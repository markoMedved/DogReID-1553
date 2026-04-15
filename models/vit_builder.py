import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import vit_b_16, ViT_B_16_Weights


def build_vit():
    weights = ViT_B_16_Weights.DEFAULT
    model = vit_b_16(weights=weights)
    model.heads = nn.Identity()
    return model


class VideoViT(nn.Module):

    def __init__(self, embedding_dim=512):
        super().__init__()

        self.vit = build_vit()

        self.embedding = nn.Linear(768, embedding_dim)

        self.bn = nn.BatchNorm1d(embedding_dim)
        self.bn.bias.requires_grad_(False)

    def forward(self, frames):

        feats = self.vit(frames)
        feats = self.embedding(feats)
        feats = self.bn(feats)

        return feats