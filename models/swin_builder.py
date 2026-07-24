import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import swin_v2_b, Swin_V2_B_Weights


class TemporalAttentionPool(nn.Module):
    """
    Learns attention weights over the temporal dimension of a video.
    """
    def __init__(self, dim):
        super().__init__()

        # --- Attention Network ---
        # Small MLP producing a scalar attention weight per frame
        self.attn = nn.Sequential(
            nn.Linear(dim, 512),
            nn.Tanh(),
            nn.Linear(512, 1),
            nn.Softmax(dim=1)  
        )

    def forward(self, x):
        # --- Compute Attention Weights ---
        weights = self.attn(x)  

        # --- Apply Weighted Pooling ---
        # Broadcasts weights across the feature dimension and sums over time
        return (x * weights).sum(dim=1)


class VideoSwin(nn.Module):
    """Model using the frozen swin_v2 backbone"""
    def __init__(self, num_classes=None, chunk_size=8):
        super().__init__()

        # --- Load Pretrained Backbone ---
        # Initializes Swin V2 Base with default ImageNet weights
        weights = Swin_V2_B_Weights.DEFAULT
        self.backbone = swin_v2_b(weights=weights)

        # Remove classification head to extract raw embeddings
        self.backbone.head = nn.Identity()

        # Swin-B standard output feature dimension
        self.dim = 1024

        # --- Memory Management ---
        self.chunk_size = chunk_size

        # --- Temporal Aggregation ---
        self.temporal_pool = TemporalAttentionPool(self.dim)

        # --- BN-Neck ---
        # Stabilizes the embedding space before metric learning
        self.bn = nn.BatchNorm1d(self.dim)

        self.bn.bias.requires_grad_(False)

    def forward(self, x):

        if x.dim() == 5:

            B, T, C, H, W = x.shape

            # Flatten temporal dimension for independent frame processing
            x = x.view(B * T, C, H, W)

            # --- Chunked Forward Pass ---
            chunks = torch.split(x, self.chunk_size, dim=0)

            feats = torch.cat(
                [self.backbone(c) for c in chunks],
                dim=0
            )

            # Reshape features back into their original video structure
            feats = feats.view(B, T, -1)

            # --- Temporal Attention Pooling ---
            # Aggregate frame-level features into a single clip embedding
            feats = self.temporal_pool(feats)

        else:
            # Standard single-image forward pass
            feats = self.backbone(x)

        # --- BN-Neck Application ---
        feats = self.bn(feats)

        # --- L2 Normalization ---
        return F.normalize(feats, p=2, dim=1)