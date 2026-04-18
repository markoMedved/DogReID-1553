import torch
import torch.nn as nn
import torch.nn.functional as F


EMBED_DIMS = {
    "vits14":     384,
    "vitb14":     768,
    "vitl14":    1024,
    "vitb14_reg": 768,
    "vitl14_reg":1024,
}

class TemporalAttentionPool(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.attn = nn.Sequential(
            nn.Linear(dim, 512),
            nn.Tanh(),
            nn.Linear(512, 1),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        # x shape: (Batch, Time, Dim)
        weights = self.attn(x) # (B, T, 1)
        # Weighted sum across the Time dimension
        return (x * weights).sum(dim=1)


class DINOv2ReID(nn.Module):
    def __init__(self, variant: str = "vitb14_reg", chunk_size: int = 32):
        super().__init__()
        self.backbone = torch.hub.load(
            "facebookresearch/dinov2", f"dinov2_{variant}"
        )
        self.chunk_size = chunk_size # Crucial for 32GB MIG slices
        D = EMBED_DIMS[variant]
        self.temporal_attn = TemporalAttentionPool(D)
        self.bn = nn.BatchNorm1d(D)
        self.bn.bias.requires_grad_(False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x can be (B, T, C, H, W) for videos or (B, C, H, W) for images
        if x.dim() == 5:
            B, T, C, H, W = x.shape
            x = x.view(B * T, C, H, W)
            
            # --- CHUNKED FORWARD ---
            # Process frames in smaller bites to stay under VRAM limit
            chunks = torch.split(x, self.chunk_size, dim=0)
            feats = torch.cat([self.backbone(c) for c in chunks], dim=0) # (B*T, D)
            
            # --- TEMPORAL ATTENTION ---
            feats = feats.view(B, T, -1)
            feats = self.temporal_attn(feats) # (B, D)
        else:
            # Standard 4D image input
            feats = self.backbone(x)

        # --- BN-NECK & L2 NORM ---
        # The normalization happens on the hypersphere created by the BN-Neck
        return F.normalize(self.bn(feats), dim=-1)