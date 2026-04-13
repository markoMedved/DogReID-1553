import torch.nn as nn
from torchvision.models import vit_b_16, ViT_B_16_Weights
import torch.nn.functional as F


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

    def __init__(self, embedding_dim=512):

        super().__init__()

        self.vit = build_vit()

        self.embedding = nn.Linear(768, embedding_dim)

        # BNNeck
        self.bn = nn.BatchNorm1d(embedding_dim)
        self.bn.bias.requires_grad_(False)

    def forward(self, clips):

        B, T, C, H, W = clips.shape

        clips = clips.view(B * T, C, H, W)

        frame_feats = self.vit(clips)  # (B*T,768)

        frame_feats = frame_feats.view(B, T, -1)

        # temporal pooling
        video_feats = frame_feats.mean(dim=1)

        feat = self.embedding(video_feats)

        feat_bn = self.bn(feat)

        # inference embedding
        return F.normalize(feat_bn, dim=1)
    




# class VideoViT(nn.Module):

#     def __init__(self, embedding_dim=512):

#         super().__init__()

#         self.vit = build_vit()

#         self.embedding = nn.Linear(768, embedding_dim)

#         self.bn = nn.BatchNorm1d(embedding_dim)

#     def forward(self, clips):

#         B, T, C, H, W = clips.shape

#         clips = clips.view(B * T, C, H, W)

#         frame_feats = self.vit(clips)   # (B*T,768)

#         frame_feats = frame_feats.view(B, T, -1)

#         video_feats = frame_feats.mean(dim=1)

#         emb = self.embedding(video_feats)

#         emb = self.bn(emb)

#         return emb