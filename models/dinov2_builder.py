import torch
import torch.nn as nn
import torch.nn.functional as F

# --- Embedding Dimensions ---
# Maps specific DINOv2 backbone variants to their output feature dimensions
EMBED_DIMS = {
    "vits14":     384,
    "vitb14":     768,
    "vitl14":    1024,
    "vitb14_reg": 768,
    "vitl14_reg":1024,
}

class TemporalAttentionPool(nn.Module):
    """
    Learns attention weights over the temporal dimension of a video.
    """

    def __init__(self, dim):
        super().__init__()

        # Simple MLP producing a scalar weight per frame
        self.attn = nn.Sequential(
            nn.Linear(dim, 512),
            nn.Tanh(),
            nn.Linear(512, 1),
            nn.Softmax(dim=1)  
        )

    def forward(self, x):
        # Compute attention weights per frame
        weights = self.attn(x)  

        # Apply weighted temporal pooling 
        return (x * weights).sum(dim=1)  


class DINOv2ReID(nn.Module):
    """
    Video ReID model utilizing a frozen DINOv2 visual backbone.
    """

    def __init__(self, variant: str = "vitb14_reg", chunk_size: int = 32):
        super().__init__()

        # --- Load Pretrained Backbone ---
        # Fetches the specified DINOv2 model from PyTorch Hub
        self.backbone = torch.hub.load(
            "facebookresearch/dinov2", f"dinov2_{variant}"
        )

        # Chunk size controls how many frames are processed simultaneously
        self.chunk_size = chunk_size

        # Retrieve embedding dimension for the selected variant
        D = EMBED_DIMS[variant]

        # --- Temporal Aggregation ---
        self.temporal_attn = TemporalAttentionPool(D)

        # --- BN-Neck ---
        # Used in ReID pipelines to stabilize the embedding space before metric learning
        self.bn = nn.BatchNorm1d(D)

        # Prevent BN bias from being updated 
        self.bn.bias.requires_grad_(False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        if x.dim() == 5:
            B, T, C, H, W = x.shape

            # Flatten temporal dimension to process frames independently through the 2D backbone
            x = x.view(B * T, C, H, W)

            # --- Chunked Forward Pass ---
            chunks = torch.split(x, self.chunk_size, dim=0)

            feats = torch.cat(
                [self.backbone(c) for c in chunks], dim=0
            )  

            # Reshape features back to their original temporal structure
            feats = feats.view(B, T, -1)

            # Aggregate temporal features into a single vector per video
            feats = self.temporal_attn(feats) 
        else:
            # Standard single-image inference
            feats = self.backbone(x)

        # --- BN-Neck Application ---
        feats = self.bn(feats)

        # --- L2 Normalization ---
        return F.normalize(feats, dim=-1)