import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import vit_b_16, ViT_B_16_Weights
import timm

class TemporalAttentionPool(nn.Module):
    """
    Learns attention weights over the temporal dimension of a video.
    """
    def __init__(self, dim):
        super().__init__()

        # --- Attention Network ---
        # Small MLP producing a scalar weight per frame
        self.attn = nn.Sequential(
            nn.Linear(dim, 512),
            nn.Tanh(),
            nn.Linear(512, 1),
            nn.Softmax(dim=1)  # Normalize across the time dimension
        )

    def forward(self, x):
        # --- Compute Attention Weights ---
        weights = self.attn(x)  

        # --- Apply Weighted Pooling ---
        # Weighted temporal aggregation
        return (x * weights).sum(dim=1)


class VideoViT(nn.Module):
    """Model using ViT as the backbone"""

    def __init__(self, chunk_size=16):
        super().__init__()

        # --- Load Pretrained Backbone ---
        # Initializes Vision Transformer with default ImageNet weights
        self.backbone = timm.create_model(
            'vit_base_patch16_224.orig_in21k', 
            pretrained=True, 
            num_classes=0
        )

        # Remove classification head to extract raw embeddings instead of logits
        #self.backbone.heads = nn.Identity()

        # Feature dimension for standard ViT-B
        self.dim = 768

        # --- Memory Management ---
        # Process frames in chunks to prevent VRAM overflow
        self.chunk_size = chunk_size

        # --- Temporal Aggregation ---
        self.temporal_pool = TemporalAttentionPool(self.dim)

        # --- BN-Neck ---
        # Stabilizes the embedding space before metric learning (used in most ReID pipelines)
        self.bn = nn.BatchNorm1d(self.dim)
        
        # Bias is typically frozen in BN-Neck implementations
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

            # Restore the temporal dimension structure
            feats = feats.view(B, T, -1)

            # --- Temporal Attention Pooling ---
            # Aggregate frame-level features into a single clip embedding
            feats = self.temporal_pool(feats)

        else:
            # Fallback for standard single-image forward pass
            feats = self.backbone(x)

        # --- BN-Neck Application ---
        feats = self.bn(feats)

        # --- L2 Normalization ---
        return F.normalize(feats, p=2, dim=1)