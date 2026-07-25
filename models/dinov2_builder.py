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
        num_classes: int = 0
    ):
        super().__init__()

        # --- Load Pretrained Backbone via torch.hub ---
        hub_name = f"dinov2_{variant}" if not variant.startswith("dinov2_") else variant
        variant_key = variant.replace("dinov2_", "")

        self.backbone = torch.hub.load("facebookresearch/dinov2", hub_name)

        self.chunk_size = chunk_size
        self.pooling_type = pooling_type.lower()
        self.num_classes = num_classes

        # Retrieve embedding dimension for the selected variant
        D = EMBED_DIMS[variant_key]

        # --- Temporal Aggregation Setup ---
        if self.pooling_type == "attn":
            self.temporal_pool = TemporalAttentionPool(D)
        elif self.pooling_type not in ["mean", "max"]:
            raise ValueError(f"Invalid pooling_type: '{pooling_type}'. Options are 'attn', 'mean', or 'max'.")

        # --- BN-Neck ---
        self.bn = nn.BatchNorm1d(D)
        self.bn.bias.requires_grad_(False)

        # --- Optional Classifier Head for Identity Loss ---
        if self.num_classes > 0:
            self.classifier = nn.Linear(D, self.num_classes, bias=False)

    def forward(self, x: torch.Tensor):
        if x.dim() == 5:
            B, T, C, H, W = x.shape

            # Flatten temporal dimension to process frames independently through the 2D backbone
            x = x.view(B * T, C, H, W)

            # --- Chunked Forward Pass ---
            chunks = torch.split(x, self.chunk_size, dim=0)
            feats = torch.cat([self.backbone(c) for c in chunks], dim=0)  

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
            feats = self.backbone(x)

        # --- BN-Neck Application ---
        feat_bn = self.bn(feats)

        # --- Training Mode with Identity Head ---
        if self.training and self.num_classes > 0:
            cls_score = self.classifier(feat_bn)
            norm_embeddings = F.normalize(feat_bn, dim=-1)
            return norm_embeddings, cls_score

        # --- Inference Mode / Metric Loss Target ---
        return F.normalize(feat_bn, dim=-1)