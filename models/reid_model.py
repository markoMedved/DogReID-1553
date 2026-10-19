import contextlib

import torch
import torch.nn as nn
import torch.nn.functional as F

from .reid_heads import BNNeckHead


def _null_context():
    return contextlib.nullcontext()


# --- Backbone Adapters ---

class DINOv2Adapter(nn.Module):
    """DINOv2 backbone loaded via torch.hub."""

    EMBED_DIMS = {
        "vits14":     384,
        "vitb14":     768,
        "vitl14":    1024,
        "vitb14_reg": 768,
        "vitl14_reg":1024,
    }

    def __init__(self, variant: str = "vitb14_reg", representation_type: str = "cls", gem_p: float = 3.0, fuse_layers: int = 1):
        super().__init__()
        hub_name = variant if variant.startswith("dinov2_") else f"dinov2_{variant}"
        self.net = torch.hub.load("facebookresearch/dinov2", hub_name)
        base_dim = self.EMBED_DIMS[variant.replace("dinov2_", "")]
        self.representation_type = (representation_type or "cls").lower()
        if self.representation_type in ("cls_mean", "cls_gem"):
            self.dim = base_dim * 2
        elif self.representation_type in ("cls", "patch_mean", "mean", "patch_gem", "gem"):
            self.dim = base_dim
        else:
            raise ValueError(f"Unknown representation_type: {representation_type!r}")

        if self.representation_type in ("patch_gem", "gem", "cls_gem"):
            self.gem_p = nn.Parameter(torch.ones(1) * float(gem_p))

        self.fuse_layers = max(1, int(fuse_layers))
        if self.fuse_layers > 1:
            self.layer_weights = nn.Parameter(torch.zeros(self.fuse_layers))

    @property
    def last_block(self):
        return self.net.blocks[-1]

    @property
    def final_norm(self):
        return self.net.norm

    @property
    def depth(self):
        return len(self.net.blocks)

    def _run_blocks(self, tokens, blocks, n_grad, return_intermediate=None):
        """Run blocks, keeping the autograd graph only for the last n_grad.

        Frozen leading blocks do not need stored activations, so running them
        under no_grad cuts activation memory without changing the result.
        """
        split = max(len(blocks) - n_grad, 0) if n_grad is not None else 0
        intermediate = []
        take_indices = set(return_intermediate) if return_intermediate is not None else set()

        if split:
            with torch.no_grad():
                for i, block in enumerate(blocks[:split]):
                    tokens = block(tokens)
                    if i in take_indices:
                        intermediate.append(tokens)
        for i, block in enumerate(blocks[split:], start=split):
            tokens = block(tokens)
            if i in take_indices:
                intermediate.append(tokens)

        if return_intermediate is not None:
            return tokens, intermediate
        return tokens

    def _aggregate_tokens(self, normed_tokens):
        cls_tok = normed_tokens[:, 0]
        if self.representation_type == "cls":
            return cls_tok

        n_reg = getattr(self.net, "num_register_tokens", 4)
        patch_toks = normed_tokens[:, 1 + n_reg:]

        if self.representation_type in ("patch_mean", "mean"):
            return patch_toks.mean(dim=1)
        if self.representation_type in ("patch_gem", "gem"):
            p = self.gem_p.clamp(min=1.0)
            return (patch_toks.clamp(min=1e-6).pow(p).mean(dim=1)).pow(1.0 / p)
        if self.representation_type == "cls_mean":
            return torch.cat([cls_tok, patch_toks.mean(dim=1)], dim=-1)
        if self.representation_type == "cls_gem":
            p = self.gem_p.clamp(min=1.0)
            patch_gem = (patch_toks.clamp(min=1e-6).pow(p).mean(dim=1)).pow(1.0 / p)
            return torch.cat([cls_tok, patch_gem], dim=-1)
        raise ValueError(f"Unknown representation_type: {self.representation_type!r}")

    def pooled(self, x, n_grad=None):
        if self.fuse_layers > 1:
            total_blocks = len(self.net.blocks)
            take_indices = list(range(total_blocks - self.fuse_layers, total_blocks))
            if n_grad is None:
                tokens = self.net.prepare_tokens_with_masks(x, None)
                _, intermediate = self._run_blocks(tokens, self.net.blocks, None, return_intermediate=take_indices)
            else:
                with torch.no_grad():
                    tokens = self.net.prepare_tokens_with_masks(x, None)
                _, intermediate = self._run_blocks(tokens, self.net.blocks, n_grad, return_intermediate=take_indices)

            layer_feats = []
            for tok in intermediate:
                normed = self.net.norm(tok)
                layer_feats.append(self._aggregate_tokens(normed))

            weights = F.softmax(self.layer_weights, dim=0)
            return sum(w * f for w, f in zip(weights, layer_feats))

        if n_grad is None and self.representation_type == "cls":
            return self.net(x)

        if n_grad is None:
            tokens = self.net.prepare_tokens_with_masks(x, None)
            tokens = self._run_blocks(tokens, self.net.blocks, None)
        else:
            with torch.no_grad():
                tokens = self.net.prepare_tokens_with_masks(x, None)
            tokens = self._run_blocks(tokens, self.net.blocks, n_grad)
        normed = self.net.norm(tokens)
        return self._aggregate_tokens(normed)


def build_backbone(cfg):
    """Instantiate the backbone adapter named by cfg.backbone."""
    name = str(getattr(cfg, "backbone", getattr(cfg, "model", "dinov2"))).lower()

    if name in ("dinov2", "baseline", "dinov2_baseline"):
        return DINOv2Adapter(
            variant=getattr(cfg, "dinov2_variant", "vitb14_reg"),
            representation_type=getattr(cfg, "representation_type", "cls"),
            gem_p=getattr(cfg, "gem_p", 3.0),
            fuse_layers=getattr(cfg, "fuse_layers", 1),
        )

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
    """Video re-ID model combining a backbone, temporal pooling and BNNeck head (BoT)."""

    def __init__(self, cfg):
        super().__init__()
        self.backbone = build_backbone(cfg)
        self.method = getattr(cfg, "reid_method", None) or "bot"
        self.chunk_size = getattr(cfg, "chunk_size", 16)
        self.num_classes = getattr(cfg, "num_classes", 0)
        self.embed_dim = self.backbone.dim

        # --- Activation Memory ---
        self.n_grad = (None if getattr(cfg, "full_finetune", False)
                       else getattr(cfg, "unfreeze_blocks", 2))

        # --- Branch Configuration (BoT Single Branch) ---
        self.n_branches = 1
        pooling = getattr(cfg, "pooling_type", "attention")
        use_bnneck = getattr(cfg, "use_bnneck", True)
        self.pools = nn.ModuleList([TemporalPool(self.embed_dim, pooling)])
        self.heads = nn.ModuleList([BNNeckHead(self.embed_dim, self.num_classes, use_bnneck=use_bnneck)])

    def _frame_features(self, x):
        """Extract features for a batch of frames."""
        n_grad = self.n_grad if self.training else 0
        return [self.backbone.pooled(x, n_grad=n_grad)]

    def forward(self, x, targets=None):
        if x.dim() == 5:
            B, T, C, H, W = x.shape
            frames = x.view(B * T, C, H, W)

            # --- Chunked Forward Pass ---
            chunks = torch.split(frames, self.chunk_size, dim=0)
            per_chunk = [self._frame_features(c) for c in chunks]

            # Restore temporal structure (B, T, D), then aggregate
            branch_feats = [
                torch.cat([chunk[i] for chunk in per_chunk], dim=0).view(B, T, -1)
                for i in range(self.n_branches)
            ]
            branch_feats = [
                pool(feat) for pool, feat in zip(self.pools, branch_feats)
            ]
        else:
            branch_feats = self._frame_features(x)

        # --- BN-Neck Application ---
        tri, ret, logit = self.heads[0](branch_feats[0])

        # --- Training Mode with Identity Head ---
        if self.training and self.num_classes > 0 and logit is not None:
            return F.normalize(tri, dim=-1), logit
        elif self.training:
            return F.normalize(tri, dim=-1)

        # --- Inference Mode ---
        return ret

