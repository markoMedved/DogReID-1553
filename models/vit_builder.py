import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import vit_b_16, ViT_B_16_Weights


class TemporalAttentionPool(nn.Module):
    """
    Learns attention weights across video frames and produces
    a single feature vector representing the whole clip.
    """

    def __init__(self, dim):
        super().__init__()

        # small MLP producing a scalar weight per frame
        self.attn = nn.Sequential(
            nn.Linear(dim, 512),
            nn.Tanh(),
            nn.Linear(512, 1),
            nn.Softmax(dim=1)  # normalize across time dimension
        )

    def forward(self, x):
        # x shape: (B, T, D)

        # compute frame importance weights
        weights = self.attn(x)  # (B, T, 1)

        # weighted temporal aggregation
        return (x * weights).sum(dim=1)


class VideoViT(nn.Module):
    """
    Video ReID model using a ViT-B/16 backbone.

    Pipeline:
    frames → ViT backbone → temporal attention pooling → BN neck → L2 normalize
    """

    def __init__(self, chunk_size=16):
        super().__init__()

        # load pretrained Vision Transformer
        weights = ViT_B_16_Weights.DEFAULT
        self.backbone = vit_b_16(weights=weights)

        # remove classification head (we want embeddings instead)
        self.backbone.heads = nn.Identity()

        # feature dimension for ViT-B
        self.dim = 768

        # process frames in chunks to reduce VRAM spikes
        self.chunk_size = chunk_size

        # temporal attention module
        self.temporal_pool = TemporalAttentionPool(self.dim)

        # BN-Neck used in most ReID pipelines
        self.bn = nn.BatchNorm1d(self.dim)
        self.bn.bias.requires_grad_(False)

    def forward(self, x):

        # expected input:
        # video: (B, T, C, H, W)
        # image: (B, C, H, W)

        if x.dim() == 5:

            B, T, C, H, W = x.shape

            # flatten temporal dimension
            x = x.view(B * T, C, H, W)

            # --- CHUNKED FORWARD ---
            # prevents GPU OOM when processing long clips
            chunks = torch.split(x, self.chunk_size, dim=0)

            feats = torch.cat(
                [self.backbone(c) for c in chunks],
                dim=0
            )

            # restore temporal dimension
            feats = feats.view(B, T, -1)

            # temporal attention pooling
            feats = self.temporal_pool(feats)

        else:
            # fallback for single images
            feats = self.backbone(x)

        # BN-Neck
        feats = self.bn(feats)

        # normalize embeddings to unit hypersphere
        return F.normalize(feats, p=2, dim=1)