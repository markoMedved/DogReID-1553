import torch
import torch.nn as nn
import torch.nn.functional as F


# Embedding dimension for each DINOv2 backbone variant
EMBED_DIMS = {
    "vits14":     384,
    "vitb14":     768,
    "vitl14":    1024,
    "vitb14_reg": 768,
    "vitl14_reg":1024,
}


class TemporalAttentionPool(nn.Module):
    """
    Learns attention weights over the temporal dimension of a video.
    Instead of averaging frames, the model learns which frames matter most.
    """

    def __init__(self, dim):
        super().__init__()

        # simple MLP producing a scalar weight per frame
        self.attn = nn.Sequential(
            nn.Linear(dim, 512),
            nn.Tanh(),
            nn.Linear(512, 1),
            nn.Softmax(dim=1)  # normalize weights across time
        )

    def forward(self, x):
        # x shape: (Batch, Time, FeatureDim)

        # compute attention weights per frame
        weights = self.attn(x)  # (B, T, 1)

        # weighted temporal pooling
        return (x * weights).sum(dim=1)  # output: (B, FeatureDim)


class DINOv2ReID(nn.Module):
    """
    Video ReID model using a frozen DINOv2 visual backbone.

    Pipeline:
    frames → DINOv2 features → temporal attention pooling → BN neck → L2 normalize
    """

    def __init__(self, variant: str = "vitb14_reg", chunk_size: int = 32):
        super().__init__()

        # load pretrained DINOv2 backbone from Torch Hub
        self.backbone = torch.hub.load(
            "facebookresearch/dinov2", f"dinov2_{variant}"
        )

        # chunk size controls how many frames are processed at once
        # prevents VRAM overflow when processing long clips
        self.chunk_size = chunk_size

        # embedding dimension depends on backbone variant
        D = EMBED_DIMS[variant]

        # temporal attention module for aggregating frame features
        self.temporal_attn = TemporalAttentionPool(D)

        # BN-Neck commonly used in ReID pipelines
        # stabilizes embedding distribution before metric learning
        self.bn = nn.BatchNorm1d(D)

        # prevent BN bias from being updated (standard ReID practice)
        self.bn.bias.requires_grad_(False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        # input can be:
        # video: (B, T, C, H, W)
        # image: (B, C, H, W)

        if x.dim() == 5:

            B, T, C, H, W = x.shape

            # flatten temporal dimension so backbone processes frames independently
            x = x.view(B * T, C, H, W)

            # --- CHUNKED FORWARD PASS ---
            # avoids VRAM spikes when processing many frames
            chunks = torch.split(x, self.chunk_size, dim=0)

            feats = torch.cat(
                [self.backbone(c) for c in chunks], dim=0
            )  # (B*T, D)

            # reshape back to temporal structure
            feats = feats.view(B, T, -1)

            # temporal attention pooling
            feats = self.temporal_attn(feats)  # (B, D)

        else:
            # standard image inference
            feats = self.backbone(x)

        # --- BN-NECK ---
        # improves metric learning performance in ReID
        feats = self.bn(feats)

        # --- L2 NORMALIZATION ---
        # embeddings lie on a unit hypersphere
        # makes cosine similarity equivalent to dot product
        return F.normalize(feats, dim=-1)