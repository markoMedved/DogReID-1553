import torch
import torch.nn as nn
import torch.nn.functional as F


class TemporalAttentionPool(nn.Module):
    """Learns attention weights over the temporal dimension of a video."""

    def __init__(self, dim):
        super().__init__()

        # Small MLP producing a scalar attention weight per frame
        self.attn = nn.Sequential(
            nn.Linear(dim, 512),
            nn.Tanh(),
            nn.Linear(512, 1),
            nn.Softmax(dim=1),
        )

    def forward(self, x):
        weights = self.attn(x)
        return (x * weights).sum(dim=1)


class MegaDescriptor(nn.Module):
    """Standalone video re-ID model on a MegaDescriptor Swin backbone.

    MegaDescriptor (Čermák et al., "WildlifeDatasets") is a Swin Transformer
    pretrained on a large multi-species wildlife re-identification corpus,
    distributed through the HuggingFace hub under the BVRA namespace and loaded
    here via timm. Unlike the ImageNet-pretrained Swin in swin_builder.py, its
    features already target the re-ID metric task, which is the relevant
    property for the unseen dog domain.

    This is the "without BoT" variant: backbone -> temporal attention pool ->
    BN neck -> L2 normalize, mirroring the other standalone builders. For the
    BNNeck + identity-loss ("with BoT") variant, set backbone='megadescriptor'
    with reid_method='bot', which routes through models/reid_model.py.

    The default variant is Swin-L at 224px; set the input transform's img_size
    to the variant's native resolution (224 here, 384 for the L-384 variant).
    """

    def __init__(self, variant: str = "hf-hub:BVRA/MegaDescriptor-L-224", chunk_size: int = 8):
        super().__init__()

        try:
            import timm
        except ImportError as exc:
            raise ImportError(
                "backbone='megadescriptor' requires timm (which pulls in "
                "huggingface_hub for the hf-hub checkpoint):\n"
                "  pip install timm huggingface_hub"
            ) from exc

        # num_classes=0 drops the classifier and returns the pooled feature.
        self.backbone = timm.create_model(variant, pretrained=True, num_classes=0)

        # Swin-L -> 1536, Swin-T -> 768; read it off the model rather than hardcode.
        self.dim = self.backbone.num_features

        # Number of frames processed simultaneously through the 2D backbone.
        self.chunk_size = chunk_size

        # --- Temporal Aggregation ---
        self.temporal_pool = TemporalAttentionPool(self.dim)

        # --- BN-Neck ---
        # Stabilizes the embedding space before metric learning.
        self.bn = nn.BatchNorm1d(self.dim)
        self.bn.bias.requires_grad_(False)

    def forward(self, x):

        if x.dim() == 5:
            B, T, C, H, W = x.shape

            # Flatten temporal dimension for independent frame processing
            x = x.view(B * T, C, H, W)

            # --- Chunked Forward Pass ---
            chunks = torch.split(x, self.chunk_size, dim=0)
            feats = torch.cat([self.backbone(c) for c in chunks], dim=0)

            # Restore temporal structure, then aggregate to one vector per clip
            feats = feats.view(B, T, -1)
            feats = self.temporal_pool(feats)

        else:
            # Standard single-image forward pass
            feats = self.backbone(x)

        # --- BN-Neck Application ---
        feats = self.bn(feats)

        # --- L2 Normalization ---
        return F.normalize(feats, p=2, dim=1)
