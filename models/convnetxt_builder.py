import torch
import torch.nn as nn
import torch.nn.functional as F
import timm


class TemporalAttentionPool(nn.Module):
    """Learns attention weights over the temporal dimension of a video."""
    def __init__(self, dim):
        super().__init__()
        self.attn = nn.Sequential(
            nn.Linear(dim, 512),
            nn.Tanh(),
            nn.Linear(512, 1),
            nn.Softmax(dim=1)  # Normalize across time dimension
        )

    def forward(self, x):
        weights = self.attn(x)  
        return (x * weights).sum(dim=1)


class VideoConvNeXt(nn.Module):
    """Model using ConvNeXt-Base (ImageNet-22k/21k) as the backbone"""

    def __init__(self, chunk_size=16):
        super().__init__()

        # --- Load Pretrained ConvNeXt Backbone ---
        # convnext_base_in22k provides official Meta ImageNet-22k weights
        self.backbone = timm.create_model(
            'convnext_base_in22k', 
            pretrained=True, 
            num_classes=0
        )

        # ConvNeXt-Base output dimension
        self.dim = 1024

        # Memory Chunking
        self.chunk_size = chunk_size

        # Temporal Pooling
        self.temporal_pool = TemporalAttentionPool(self.dim)

        # BN-Neck
        self.bn = nn.BatchNorm1d(self.dim)
        self.bn.bias.requires_grad_(False)

    def forward(self, x):
        if x.dim() == 5:
            B, T, C, H, W = x.shape
            x = x.view(B * T, C, H, W)

            # --- Chunked Forward Pass ---
            chunks = torch.split(x, self.chunk_size, dim=0)
            feats = torch.cat(
                [self.backbone(c) for c in chunks],
                dim=0
            )

            # Restore temporal dimension: (B, T, D)
            feats = feats.view(B, T, -1)

            # --- Temporal Attention Pooling ---
            feats = self.temporal_pool(feats)

        else:
            # Fallback for single-frame forward pass
            feats = self.backbone(x)

        # --- BN-Neck & L2 Normalization ---
        feats = self.bn(feats)
        return F.normalize(feats, p=2, dim=1)