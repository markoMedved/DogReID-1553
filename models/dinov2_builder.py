import torch
import torch.nn as nn
import torch.nn.functional as F

# --- Embedding Dimensions ---
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
    """
    def __init__(self, dim):
        super().__init__()
        self.attn = nn.Sequential(
            nn.Linear(dim, 512),
            nn.Tanh(),
            nn.Linear(512, 1),
            nn.Softmax(dim=1)  
        )

    def forward(self, x):
        weights = self.attn(x)  
        return (x * weights).sum(dim=1)  


class DINOv2ReID(nn.Module):
    """
    Video ReID model utilizing a DINOv2 visual backbone via PyTorch Hub.
    Supports dual-loss strategy (Triplet Margin + Identity CrossEntropy).
    """

    def __init__(
        self, 
        variant: str = "vitb14_reg", 
        num_classes: int = 0, 
        chunk_size: int = 32
    ):
        super().__init__()

        # --- Load Pretrained Backbone via torch.hub ---
        hub_name = f"dinov2_{variant}" if not variant.startswith("dinov2_") else variant
        variant_key = variant.replace("dinov2_", "")

        self.backbone = torch.hub.load("facebookresearch/dinov2", hub_name)

        self.chunk_size = chunk_size
        self.num_classes = num_classes

        # Retrieve embedding dimension
        D = EMBED_DIMS[variant_key]

        # --- Temporal Aggregation ---
        self.temporal_attn = TemporalAttentionPool(D)

        # --- BN-Neck ---
        self.bn = nn.BatchNorm1d(D)
        self.bn.bias.requires_grad_(False)

        # --- Classifier Head for Identity Loss ---
        if self.num_classes > 0:
            self.classifier = nn.Linear(D, self.num_classes, bias=False)

    def forward(self, x: torch.Tensor):
        if x.dim() == 5:
            B, T, C, H, W = x.shape

            # Flatten temporal dimension to process frames through 2D backbone
            x = x.view(B * T, C, H, W)

            # --- Chunked Forward Pass ---
            chunks = torch.split(x, self.chunk_size, dim=0)
            feats = torch.cat([self.backbone(c) for c in chunks], dim=0)  

            # Reshape features back to temporal structure
            feats = feats.view(B, T, -1)

            # Aggregate temporal features
            feats = self.temporal_attn(feats) 
        else:
            # Standard single-image inference
            feats = self.backbone(x)

        # --- BN-Neck Application ---
        feat_bn = self.bn(feats)

        # --- Training Mode with Identity Head ---
        if self.training and self.num_classes > 0:
            cls_score = self.classifier(feat_bn)
            norm_embeddings = F.normalize(feat_bn, dim=-1)
            return norm_embeddings, cls_score

        # --- Inference Mode / Metric Loss Target ---
        return F.normalize(feat_bn, dim=-1)