import torch
import torch.nn as nn
import torch.nn.functional as F

# --- Embedding Dimensions ---
# Maps specific DINOv2 backbone variants to their output feature dimensions
EMBED_DIMS = {
    "vits14": 384,
    "vitb14": 768,
    "vitl14": 1024,
    "vitb14_reg": 768,
    "vitl14_reg": 1024,
}

class TemporalAttentionPool(nn.Module):
    """
    Learns attention weights over the temporal dimension of a video.
    """
    def __init__(self, dim):
        super().__init__()
        self.attn = nn.Sequential(
            nn.Linear(dim, 512),
            nn.Tanh(),
            nn.Linear(512, 1),
            nn.Softmax(dim=1)  
        )

    def forward(self, x):
        weights = self.attn(x)  
        return (x * weights).sum(dim=1)  


class DINOv2ReID(nn.Module):
    """
    Unified Video/Image ReID model utilizing a DINOv2 visual backbone.
    Supports flexible pooling types ('attn', 'mean', 'max'), chunked VRAM processing,
    and optional identity classification head for dual-loss setups.
    """

    def __init__(
        self, 
        variant: str = "vitb14_reg", 
        chunk_size: int = 32, 
        pooling_type: str = "attn",
        num_classes: int = 0,
        representation_type: str = "cls_mean",
        gem_p: float = 3.0,
        use_bnneck: bool = True,
    ):
        super().__init__()

        # --- Load Pretrained Backbone via torch.hub ---
        hub_name = f"dinov2_{variant}" if not variant.startswith("dinov2_") else variant
        variant_key = variant.replace("dinov2_", "")

        self.backbone = torch.hub.load("facebookresearch/dinov2", hub_name)

        self.chunk_size = chunk_size
        p_type = pooling_type.lower()
        self.pooling_type = "attn" if p_type in ["attention", "attn"] else p_type
        self.num_classes = num_classes
        self.representation_type = representation_type.lower()
        self.gem_p = gem_p
        self.use_bnneck = use_bnneck

        # Retrieve base feature dimension for the selected variant
        base_d = EMBED_DIMS[variant_key]

        # Calculate final embedding dimension based on representation strategy
        if self.representation_type in ("cls_mean", "gem", "cls_gem"):
            D = base_d * 2
        elif self.representation_type == "cls":
            D = base_d
        elif self.representation_type in ("patch_mean", "mean"):
            D = base_d
        else:
            raise ValueError(
                f"Invalid representation_type: '{representation_type}'. "
                "Options are 'cls_gem', 'gem', 'cls_mean', 'cls', or 'patch_mean'."
            )
        self.embed_dim = D

        # --- Temporal Aggregation Setup ---
        if self.pooling_type == "attn":
            self.temporal_attn = TemporalAttentionPool(D)
        elif self.pooling_type not in ["mean", "max"]:
            raise ValueError(f"Invalid pooling_type: '{pooling_type}'. Options are 'attn', 'mean', or 'max'.")

        # --- BN-Neck ---
        if self.use_bnneck:
            self.bn = nn.BatchNorm1d(D)
            self.bn.bias.requires_grad_(False)
        else:
            self.bn = nn.Identity()

        # --- Optional Classifier Head for Identity Loss ---
        if self.num_classes > 0:
            self.classifier = nn.Linear(D, self.num_classes, bias=False)

    @property
    def temporal_pool(self):
        return getattr(self, "temporal_attn", None)

    def _extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract spatial / token representations from DINOv2.
        - 'cls_mean': Strategy 1 - Concat [CLS] with spatial average of patch tokens [B, 2*D]
        - 'gem': Strategy 2 - Concat [CLS] with Generalized Mean (GeM) of patch tokens [B, 2*D]
        - 'cls': Pure [CLS] token [B, D]
        - 'patch_mean': Pure average of patch tokens [B, D]
        """
        if self.representation_type == "cls":
            return self.backbone(x)

        out = self.backbone.forward_features(x)
        cls_tok = out["x_norm_clstoken"]
        patch_toks = out["x_norm_patchtokens"]

        if self.representation_type == "cls_mean":
            patch_mean = patch_toks.mean(dim=1)
            return torch.cat([cls_tok, patch_mean], dim=-1)
        elif self.representation_type in ("gem", "cls_gem"):
            eps = 1e-6
            p = self.gem_p
            patch_gem = (patch_toks.clamp(min=eps).pow(p).mean(dim=1)).pow(1.0 / p)
            return torch.cat([cls_tok, patch_gem], dim=-1)
        elif self.representation_type in ("patch_mean", "mean"):
            return patch_toks.mean(dim=1)
        else:
            raise ValueError(f"Unknown representation_type: {self.representation_type}")

    def forward(self, x: torch.Tensor, targets: torch.Tensor = None, **kwargs):
        if x.dim() == 5:
            B, T, C, H, W = x.shape

            # Flatten temporal dimension to process frames independently through the 2D backbone
            x = x.view(B * T, C, H, W)

            # --- Chunked Forward Pass ---
            chunks = torch.split(x, self.chunk_size, dim=0)
            feats = torch.cat([self._extract_features(c) for c in chunks], dim=0)  

            # Reshape features back to their original temporal structure [B, T, D]
            feats = feats.view(B, T, -1)

            # --- Aggregate temporal features into a single vector per video ---
            if self.pooling_type == "attn":
                feats = self.temporal_pool(feats)
            elif self.pooling_type == "mean":
                feats = feats.mean(dim=1)
            elif self.pooling_type == "max":
                feats = feats.max(dim=1)[0]
        else:
            # Standard single-image inference [B, C, H, W]
            feats = self._extract_features(x)

        # --- BN-Neck Application ---
        feat_bn = self.bn(feats)

        # --- Training Mode with Identity Head ---
        if self.training and self.num_classes > 0:
            cls_score = self.classifier(feat_bn)
            norm_embeddings = F.normalize(feat_bn, dim=-1)
            return norm_embeddings, cls_score

        # --- Inference Mode / Metric Loss Target ---
        return F.normalize(feat_bn, dim=-1)