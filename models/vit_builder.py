import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import vit_b_16, ViT_B_16_Weights

def build_vit():
    weights = ViT_B_16_Weights.DEFAULT
    model = vit_b_16(weights=weights)
    model.heads = nn.Identity()
    return model

class VideoViT(nn.Module):
    def __init__(self, freeze_backbone=False):
        super().__init__()
        self.vit = build_vit()
        if freeze_backbone:
            for p in self.vit.parameters():
                p.requires_grad = False

    def forward(self, x):
        # Case 1: Video Input (B, T, C, H, W)
        if x.dim() == 5:
            B, T, C, H, W = x.shape
            # Flatten to (B*T, C, H, W)
            x = x.view(B * T, C, H, W)
            feats = self.vit(x)          # (B*T, 768)
            feats = feats.view(B, T, -1) # (B, T, 768)
            return feats.mean(dim=1)     # (B, 768) - Temporal Mean Pool

        # Case 2: Standard Image/Frame Input (N, C, H, W)
        return self.vit(x)