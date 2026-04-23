import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import swin_v2_b, Swin_V2_B_Weights


class TemporalAttentionPool(nn.Module):
    """
    Learns attention weights over video frames and produces
    a single feature vector representing the entire clip.
    """

    def __init__(self, dim):
        super().__init__()

        # small MLP producing a scalar attention weight per frame
        self.attn = nn.Sequential(
            nn.Linear(dim, 512),
            nn.Tanh(),
            nn.Linear(512, 1),
            nn.Softmax(dim=1)  # normalize weights across time dimension
        )

    def forward(self, x):
        # x shape: (B, T, Dim)

        # compute frame importance weights
        weights = self.attn(x)  # (B, T, 1)

        # weighted temporal pooling
        return (x * weights).sum(dim=1)


class VideoSwin(nn.Module):
    """
    Video ReID model using a Swin V2 backbone.

    Pipeline:
    frames → Swin backbone → temporal attention pooling → BN neck → L2 normalize
    """

    def __init__(self, num_classes=None, chunk_size=8):
        super().__init__()

        # load pretrained Swin V2 Base model
        weights = Swin_V2_B_Weights.DEFAULT
        self.backbone = swin_v2_b(weights=weights)

        # remove classification head since we want embeddings
        self.backbone.head = nn.Identity()

        # Swin-B feature dimension
        self.dim = 1024

        # number of frames processed at once (helps avoid VRAM overflow)
        self.chunk_size = chunk_size

        # temporal attention pooling for aggregating frame features
        self.temporal_pool = TemporalAttentionPool(self.dim)

        # BN-Neck commonly used in ReID pipelines
        # stabilizes embedding space before metric learning
        self.bn = nn.BatchNorm1d(self.dim)

        # bias is typically frozen in BN-Neck
        self.bn.bias.requires_grad_(False)

    def forward(self, x):

        # expected input shapes:
        # video: (Batch, Time, Channels, Height, Width)
        # image: (Batch, Channels, Height, Width)

        if x.dim() == 5:

            B, T, C, H, W = x.shape

            # flatten temporal dimension so backbone processes frames independently
            x = x.view(B * T, C, H, W)

            # --- CHUNKED FORWARD ---
            # Swin-V2-B is memory intensive.
            # Process frames in smaller chunks to avoid GPU OOM.
            chunks = torch.split(x, self.chunk_size, dim=0)

            feats = torch.cat(
                [self.backbone(c) for c in chunks],
                dim=0
            )

            # reshape back into video structure
            feats = feats.view(B, T, -1)

            # --- TEMPORAL ATTENTION POOLING ---
            # aggregate frame-level features into a single clip embedding
            feats = self.temporal_pool(feats)

        else:
            # standard single-image forward pass
            feats = self.backbone(x)

        # --- BN-NECK ---
        feats = self.bn(feats)

        # --- L2 NORMALIZATION ---
        # ensures embeddings lie on unit hypersphere
        return F.normalize(feats, p=2, dim=1)