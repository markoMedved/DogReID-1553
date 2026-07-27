import torch
import torch.nn as nn
import torch.nn.functional as F


class TemporalAttentionPool(nn.Module):
    """
    Learns attention weights over the temporal dimension of a video.
    """

    def __init__(self, dim):
        super().__init__()

        # Small MLP producing a scalar attention weight per frame
        self.attn = nn.Sequential(
            nn.Linear(dim, 512),
            nn.Tanh(),
            nn.Linear(512, 1),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        weights = self.attn(x)
        return (x * weights).sum(dim=1)


class MiewIDReID(nn.Module):
    """
    Video ReID model wrapping the MiewID wildlife re-identification backbone.
    """

    HF_REPO = "conservationxlabs/miewid-msv3"

    def __init__(self, chunk_size: int = 16):
        super().__init__()

        # --- Load Pretrained Backbone ---
        # MiewID ships a custom forward that returns pooled embeddings, so
        # trust_remote_code is required. transformers is imported lazily so that
        # merely importing this module (e.g. through models.model_factory or the
        # evaluation script) does not force a transformers dependency on runs
        # that use a different backbone.
        try:
            from transformers import AutoModel
        except ImportError as exc:
            raise ImportError(
                "backbone='miewid' requires transformers:\n"
                "  pip install transformers"
            ) from exc

        self.backbone = AutoModel.from_pretrained(
            self.HF_REPO, trust_remote_code=True
        )

        self.chunk_size = chunk_size

        # --- Infer Embedding Dimension ---
        # The checkpoint's embedding size is not exposed directly, so we probe
        # it with a dummy forward. Kept on CPU here to avoid CUDA init in
        # __init__; the model is moved to the target device by the caller.
        with torch.no_grad():
            dummy = torch.zeros(1, 3, 224, 224)
            feats = self.backbone(dummy)
            if isinstance(feats, (tuple, list)):
                feats = feats[0]
        D = feats.shape[-1]
        self.dim = D

        # --- Temporal Aggregation ---
        self.temporal_pool = TemporalAttentionPool(D)

        # --- BN-Neck ---
        self.bn = nn.BatchNorm1d(D)
        self.bn.bias.requires_grad_(False)

    def _extract(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.backbone(x)
        if isinstance(feats, (tuple, list)):
            feats = feats[0]
        return feats

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        if x.dim() == 5:
            B, T, C, H, W = x.shape

            # Flatten temporal dimension for independent frame processing
            x = x.view(B * T, C, H, W)

            # --- Chunked Forward Pass ---
            chunks = torch.split(x, self.chunk_size, dim=0)
            feats = torch.cat([self._extract(c) for c in chunks], dim=0)

            # Restore temporal structure and aggregate
            feats = feats.view(B, T, -1)
            feats = self.temporal_pool(feats)
        else:
            feats = self._extract(x)

        feats = self.bn(feats)
        return F.normalize(feats, p=2, dim=1)
