import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import vit_b_16, ViT_B_16_Weights
import torch


def build_vit():
    """
    Builds ViT-B/16 pre-trained model with classifier removed (Identity)
    """
    weights = ViT_B_16_Weights.DEFAULT
    model = vit_b_16(weights=weights)
    
    # Remove classification head -> return embeddings
    model.heads = nn.Identity()

    return model


class VideoViT(nn.Module):

    def __init__(self):
        super().__init__()
        self.vit = build_vit()

    def forward(self, frames):
        feats = self.vit(frames)  # (N, 768)
        return feats