import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

class TemporalAttentionPool(nn.Module):
    """
    Learns attention weights over the temporal dimension of a video.
    """
    def __init__(self, dim):
        super().__init__()

        # --- Attention Network ---
        self.attn = nn.Sequential(
            nn.Linear(dim, 512),
            nn.Tanh(),
            nn.Linear(512, 1),
            nn.Softmax(dim=1)  # Normalize across the time dimension
        )

    def forward(self, x):
        # Broadcasts weights across the feature dimension and sums over time
        weights = self.attn(x)  
        return (x * weights).sum(dim=1)


class VideoSwin(nn.Module):
    """Model using SwinV2-Base (ImageNet-21k/22k) as the backbone"""

    def __init__(self, chunk_size=8):
        super().__init__()

        # --- Load Pretrained Backbone (ImageNet-22k / 21k) ---
        # num_classes=0 strips the final classification head and outputs raw 1024-dim features
        self.backbone = timm.create_model(
            'swinv2_base_window12_192', 
            pretrained=True, 
            num_classes=0
        )

        # SwinV2-Base feature dimension
        self.dim = 1024

        # --- Memory Management ---
        # Smaller chunk_size recommended for Swin due to window attention memory
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

            # Reshape features back into original video structure: (B, T, D)
            feats = feats.view(B, T, -1)

            # --- Temporal Attention Pooling ---
            feats = self.temporal_pool(feats)

        else:
            # Standard single-image forward pass
            feats = self.backbone(x)

        # --- BN-Neck Application ---
        feats = self.bn(feats)

        # --- L2 Normalization ---
        return F.normalize(feats, p=2, dim=1)