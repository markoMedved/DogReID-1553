import torch
import torch.nn as nn
from .vit_backbone import ViTBackbone


class VideoViTReID(nn.Module):

    def __init__(self, embedding_dim=512):

        super().__init__()

        self.backbone = ViTBackbone()

        self.embedding = nn.Linear(self.backbone.out_dim, embedding_dim)

        self.bn = nn.BatchNorm1d(embedding_dim)

    def forward(self, clips):

        B, T, C, H, W = clips.shape

        clips = clips.view(B * T, C, H, W)

        frame_features = self.backbone(clips)

        frame_features = frame_features.view(B, T, -1)

        video_features = frame_features.mean(dim=1)

        embedding = self.embedding(video_features)

        embedding = self.bn(embedding)

        return embedding