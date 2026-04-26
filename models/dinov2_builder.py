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
    Instead of simply averaging frames, the model learns which frames contain the most useful information.
    """

    def __init__(self, dim):
        super().__init__()

        # Simple MLP producing a scalar weight per frame
        self.attn = nn.Sequential(
            nn.Linear(dim, 512),
            nn.Tanh(),
            nn.Linear(512, 1),
            nn.Softmax(dim=1)  # Normalize weights across the temporal dimension
        )

    def forward(self, x):
        # x shape: (Batch, Time, FeatureDim)

        # Compute attention weights per frame
        weights = self.attn(x)  # Output shape: (B, T, 1)

        # Apply weighted temporal pooling via broadcasting
        return (x * weights).sum(dim=1)  # Output shape: (B, FeatureDim)


class DINOv2ReID(nn.Module):
    """
    Video ReID model utilizing a frozen DINOv2 visual backbone.

    Pipeline:
    frames -> DINOv2 features -> temporal attention pooling -> BN neck -> L2 normalize
    """

    def __init__(self, variant: str = "vitb14_reg", chunk_size: int = 32):
        super().__init__()

        # --- Load Pretrained Backbone ---
        # Fetches the specified DINOv2 model from PyTorch Hub
        self.backbone = torch.hub.load(
            "facebookresearch/dinov2", f"dinov2_{variant}"
        )

        # Chunk size controls how many frames are processed simultaneously
        # Crucial for preventing Out-Of-Memory (OOM) errors during long video inference
        self.chunk_size = chunk_size

        # Retrieve embedding dimension for the selected variant
        D = EMBED_DIMS[variant]

        # --- Temporal Aggregation ---
        self.temporal_attn = TemporalAttentionPool(D)

        # --- BN-Neck ---
        # Commonly used in ReID pipelines to stabilize the embedding space before metric learning
        self.bn = nn.BatchNorm1d(D)

        # Prevent BN bias from being updated (standard ReID practice)
        self.bn.bias.requires_grad_(False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        # Input dimensionalities:
        # Video: (B, T, C, H, W)
        # Image: (B, C, H, W)

        if x.dim() == 5:
            B, T, C, H, W = x.shape

            # Flatten temporal dimension to process frames independently through the 2D backbone
            x = x.view(B * T, C, H, W)

            # --- Chunked Forward Pass ---
            # Splits the batch to avoid VRAM spikes
            chunks = torch.split(x, self.chunk_size, dim=0)

            feats = torch.cat(
                [self.backbone(c) for c in chunks], dim=0
            )  # Output shape: (B*T, D)

            # Reshape features back to their original temporal structure
            feats = feats.view(B, T, -1)

            # Aggregate temporal features into a single vector per video
            feats = self.temporal_attn(feats)  # Output shape: (B, D)

        else:
            # Standard single-image inference
            feats = self.backbone(x)

        # --- BN-Neck Application ---
        # Smooths the feature distribution for improved metric learning
        feats = self.bn(feats)

        # --- L2 Normalization ---
        # Projects embeddings onto a unit hypersphere, making cosine similarity equivalent to the dot product
        return F.normalize(feats, dim=-1)