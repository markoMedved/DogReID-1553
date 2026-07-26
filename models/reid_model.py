import contextlib

import torch
import torch.nn as nn
import torch.nn.functional as F

from .reid_heads import JPM, BNNeckHead


def _null_context():
    return contextlib.nullcontext()


# --- Backbone Adapters ---
# Each adapter exposes a uniform interface over a different backbone family:
#   pooled(x)      -> (N, D) frame-level feature
#   penultimate(x) -> (N, 1+L, D) token sequence before the final block (JPM only)

class DINOv2Adapter(nn.Module):
    """DINOv2 backbone loaded via torch.hub."""

    EMBED_DIMS = {
        "vits14":     384,
        "vitb14":     768,
        "vitl14":    1024,
        "vitb14_reg": 768,
        "vitl14_reg":1024,
    }

    def __init__(self, variant: str = "vitb14_reg"):
        super().__init__()
        hub_name = variant if variant.startswith("dinov2_") else f"dinov2_{variant}"
        self.net = torch.hub.load("facebookresearch/dinov2", hub_name)
        self.dim = self.EMBED_DIMS[variant.replace("dinov2_", "")]

    @property
    def last_block(self):
        return self.net.blocks[-1]

    @property
    def final_norm(self):
        return self.net.norm

    @property
    def depth(self):
        return len(self.net.blocks)

    def _run_blocks(self, tokens, blocks, n_grad):
        """Run blocks, keeping the autograd graph only for the last n_grad.

        Frozen leading blocks do not need stored activations, so running them
        under no_grad cuts activation memory without changing the result.
        """
        split = max(len(blocks) - n_grad, 0) if n_grad is not None else 0

        if split:
            with torch.no_grad():
                for block in blocks[:split]:
                    tokens = block(tokens)
        for block in blocks[split:]:
            tokens = block(tokens)
        return tokens

    def penultimate(self, x, n_grad=None):
        # The final block is applied by JPM, so one fewer trainable block here
        n_grad_here = None if n_grad is None else max(n_grad - 1, 0)
        with torch.no_grad() if n_grad_here == 0 else _null_context():
            tokens = self.net.prepare_tokens_with_masks(x, None)
        return self._run_blocks(tokens, self.net.blocks[:-1], n_grad_here)

    def pooled(self, x, n_grad=None):
        if n_grad is None:
            return self.net(x)

        with torch.no_grad():
            tokens = self.net.prepare_tokens_with_masks(x, None)
        tokens = self._run_blocks(tokens, self.net.blocks, n_grad)
        return self.net.norm(tokens)[:, 0]


class TimmAdapter(nn.Module):
    """Backbones loaded from timm. Provides pooled features only."""

    def __init__(self, name: str):
        super().__init__()
        import timm
        self.net = timm.create_model(name, pretrained=True, num_classes=0)
        self.dim = self.net.num_features

    def penultimate(self, x, n_grad=None):
        raise NotImplementedError(
            "This backbone exposes no flat token sequence. "
            "reid_method='transreid' requires a ViT-family backbone."
        )

    def pooled(self, x, n_grad=None):
        return self.net(x)


TIMM_BACKBONES = {
    "vit":       "vit_base_patch16_224.augreg_in21k",
    "swin":      "swinv2_base_window12_192.ms_in22k",
    "convnext":  "convnext_base.fb_in22k",
    "convnetxt": "convnext_base.fb_in22k",
}


def build_backbone(cfg):
    """Instantiate the backbone adapter named by cfg.backbone."""
    name = getattr(cfg, "backbone", getattr(cfg, "model", "dinov2"))

    if name == "dinov2":
        return DINOv2Adapter(getattr(cfg, "dinov2_variant", "vitb14_reg"))

    if name in TIMM_BACKBONES:
        return TimmAdapter(TIMM_BACKBONES[name])

    raise ValueError(f"Unknown backbone requested: {name!r}")


# --- Temporal Aggregation ---

class TemporalPool(nn.Module):
    """Aggregates frame features into a single clip-level feature."""

    def __init__(self, dim: int, mode: str = "attention"):
        super().__init__()
        if mode not in ("attention", "mean", "max"):
            raise ValueError(f"Unknown pooling_type: {mode!r}")
        self.mode = mode

        if mode == "attention":
            self.attn = nn.Sequential(
                nn.Linear(dim, 512),
                nn.Tanh(),
                nn.Linear(512, 1),
                nn.Softmax(dim=1),
            )

    def forward(self, x):
        if self.mode == "attention":
            return (x * self.attn(x)).sum(dim=1)
        if self.mode == "mean":
            return x.mean(dim=1)
        return x.max(dim=1).values


# --- Model ---

class VideoReID(nn.Module):
    """Video re-ID model combining a backbone, temporal pooling and BNNeck heads.

    reid_method:
        "bot"       BNNeck baseline, one branch (Luo et al., 2020).
        "transreid" Jigsaw Patch Module, one global plus k local branches
                    (He et al., 2021). Requires a ViT-family backbone.
    """

    def __init__(self, cfg):
        super().__init__()
        self.backbone = build_backbone(cfg)
        self.method = getattr(cfg, "reid_method", "bot")
        self.chunk_size = getattr(cfg, "chunk_size", 16)
        self.num_classes = getattr(cfg, "num_classes", 0)
        self.embed_dim = self.backbone.dim

        # --- Activation Memory ---
        # Under partial fine-tuning the leading blocks are frozen, so their
        # activations need not be retained. n_grad = None means "keep
        # everything", which is what full fine-tuning requires.
        self.n_grad = (None if getattr(cfg, "full_finetune", False)
                       else getattr(cfg, "unfreeze_blocks", 2))

        # --- Branch Configuration ---
        if self.method == "transreid":
            if not (hasattr(self.backbone, "last_block") and hasattr(self.backbone, "final_norm")):
                raise ValueError(
                    f"reid_method='transreid' requires a ViT-family backbone exposing its "
                    f"final block; {type(self.backbone).__name__} does not. "
                    f"Use backbone='dinov2' or set reid_method='bot'."
                )
            self.jpm = JPM(
                self.backbone.last_block,
                self.backbone.final_norm,
                k=getattr(cfg, "jpm_parts", 4),
                shift=getattr(cfg, "jpm_shift", 5),
                shuffle_groups=getattr(cfg, "jpm_shuffle_groups", 2),
            )
            self.n_branches = self.jpm.k + 1
        elif self.method == "bot":
            self.jpm = None
            self.n_branches = 1
        else:
            raise ValueError(f"Unknown reid_method: {self.method!r}")

        # --- Per-Branch Pooling and Heads ---
        # Branch heads are kept separate; sharing them collapses the local features.
        pooling = getattr(cfg, "pooling_type", "attention")
        self.pools = nn.ModuleList(
            [TemporalPool(self.embed_dim, pooling) for _ in range(self.n_branches)]
        )
        self.heads = nn.ModuleList(
            [BNNeckHead(self.embed_dim, self.num_classes) for _ in range(self.n_branches)]
        )

    def _frame_features(self, x):
        """Extract one feature per branch for a batch of frames."""
        n_grad = self.n_grad if self.training else 0
        if self.jpm is None:
            return [self.backbone.pooled(x, n_grad=n_grad)]
        return self.jpm(self.backbone.penultimate(x, n_grad=n_grad))

    def forward(self, x):
        if x.dim() == 5:
            B, T, C, H, W = x.shape

            # Flatten temporal dimension to process frames through the 2D backbone
            frames = x.view(B * T, C, H, W)

            # --- Chunked Forward Pass ---
            chunks = torch.split(frames, self.chunk_size, dim=0)
            per_chunk = [self._frame_features(c) for c in chunks]

            # Regroup by branch, restore temporal structure, then aggregate
            branch_feats = [
                torch.cat([chunk[b] for chunk in per_chunk], dim=0).view(B, T, -1)
                for b in range(self.n_branches)
            ]
            branch_feats = [pool(f) for pool, f in zip(self.pools, branch_feats)]
        else:
            # Standard single-image inference
            branch_feats = self._frame_features(x)

        # --- BN-Neck Application ---
        triplet_feats, retrieval_feats, logits = [], [], []
        for head, feat in zip(self.heads, branch_feats):
            tri, ret, logit = head(feat)
            triplet_feats.append(tri)
            retrieval_feats.append(ret)
            if logit is not None:
                logits.append(logit)

        # --- Training Mode with Identity Head ---
        # Triplet loss uses the pre-BN global feature; identity loss uses the logits.
        if self.training and self.num_classes > 0:
            embeddings = F.normalize(triplet_feats[0], dim=-1)
            cls_score = logits[0] if len(logits) == 1 else torch.stack(logits, dim=0).mean(0)
            return embeddings, cls_score

        # --- Inference Mode ---
        # TransReID concatenates the global and local branch features.
        if self.n_branches > 1:
            return F.normalize(torch.cat(retrieval_feats, dim=-1), dim=-1)
        return retrieval_feats[0]
