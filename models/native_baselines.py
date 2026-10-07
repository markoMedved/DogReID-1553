"""Native implementations of ARBase, AGW, SBS, and MGN Re-ID baselines.

Each model is implemented in pure PyTorch and directly inherits the original
architectures, hyperparameters, and loss formulations:
- ARBase: ResNet50-IBN-a + GlobalAvgPool + BNNeck + Triplet(0.3) + Label-smoothed CE
- AGW: ResNet50-IBN-a + NonLocal + GeM(p=3) + BNNeck + Weighted Regularized Soft-margin Triplet + CE
- SBS: ResNet50-IBN-a + NonLocal + GeMP(trainable p) + BNNeck + CircleSoftmax + Soft-margin Triplet (post-BN)
- MGN: ResNet50-IBN-a + 3 branches (stride 2 & 1) + 8-part Multi-Granularity stripe pooling + 8 BNNeck heads
"""

import copy
import torch
import torch.nn as nn
import torch.nn.functional as F

from .losses import triplet_loss, cross_entropy_loss, CircleSoftmax
from .reid_heads import BNNeckHead
from .reid_model import TemporalPool
from .resnet_ibn import Bottleneck, ResNet50_IBN_a


# --- Spatial Pooling Layers ---

class GeneralizedMeanPooling(nn.Module):
    """Generalized Mean Pooling (GeM) with fixed exponent p.

    Radenović et al., 'Fine-tuning CNN Image Retrieval with No Human Annotation', TPAMI 2018.
    """

    def __init__(self, norm: float = 3.0, eps: float = 1e-6):
        super().__init__()
        self.p = float(norm)
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.clamp(min=self.eps).pow(self.p)
        return F.adaptive_avg_pool2d(x, (1, 1)).pow(1.0 / self.p)

    def extra_repr(self) -> str:
        return f"p={self.p}, eps={self.eps}"


class GeneralizedMeanPoolingP(nn.Module):
    """Generalized Mean Pooling (GeM) with trainable exponent p (SBS)."""

    def __init__(self, norm: float = 3.0, eps: float = 1e-6):
        super().__init__()
        self.p = nn.Parameter(torch.ones(1) * float(norm))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        p = self.p.clamp(min=1e-6)
        x = x.clamp(min=self.eps).pow(p)
        return F.adaptive_avg_pool2d(x, (1, 1)).pow(1.0 / p)

    def extra_repr(self) -> str:
        return f"p=trainable, eps={self.eps}"


# --- CircleSoftmax Head for SBS ---

class CircleSoftmaxHead(nn.Module):
    """BNNeck + CircleSoftmax classifier head for SBS.

    In SBS, metric learning is applied to features AFTER the BN bottleneck (NECK_FEAT: after).
    """

    def __init__(self, dim: int, num_classes: int = 0, scale: float = 64.0, margin: float = 0.35):
        super().__init__()
        self.dim = dim
        self.num_classes = num_classes

        self.bottleneck = nn.BatchNorm1d(dim)
        self.bottleneck.bias.requires_grad_(False)
        nn.init.constant_(self.bottleneck.weight, 1.0)
        nn.init.constant_(self.bottleneck.bias, 0.0)

        if num_classes > 0:
            self.weight = nn.Parameter(torch.Tensor(num_classes, dim))
            nn.init.normal_(self.weight, std=0.01)
            self.cls_layer = CircleSoftmax(num_classes, scale=scale, margin=margin)
        else:
            self.weight = None
            self.cls_layer = None

    def forward(self, feat: torch.Tensor, targets: torch.Tensor = None):
        """Returns:

        - feat_tri: feature for metric loss (feat_bn in SBS)
        - feat_ret: normalized retrieval feature
        - cls_outputs: modulated logits or raw cosine logits
        """
        feat_bn = self.bottleneck(feat)
        feat_ret = F.normalize(feat_bn, dim=-1)

        if not self.training:
            return feat_ret

        cls_outputs = None
        if self.num_classes > 0 and self.weight is not None:
            # Cosine similarity between normalized feature and normalized weights
            logits = F.linear(F.normalize(feat_bn, dim=-1), F.normalize(self.weight, dim=-1))
            if targets is not None:
                cls_outputs = self.cls_layer(logits, targets)
            else:
                cls_outputs = logits.mul(self.cls_layer.s)

        # In SBS, triplet loss is computed on feat_bn (after BNNeck)
        return feat_bn, feat_ret, cls_outputs


# --- Helper for chunked video forward ---

def _chunked_backbone_forward(backbone, x: torch.Tensor, chunk_size: int = 64) -> torch.Tensor:
    """Run frames through backbone in chunks to preserve GPU memory."""
    if x.dim() == 5:
        B, T, C, H, W = x.shape
        frames = x.transpose(0, 1).reshape(T * B, C, H, W)
        chunks = torch.split(frames, chunk_size, dim=0)
        per_chunk = [backbone(c) for c in chunks]
        return torch.cat(per_chunk, dim=0), B, T
    else:
        out = backbone(x)
        return out, x.size(0), 1


# --- 1. ARBase ---

class NativeARBase(nn.Module):
    """Native ARBase: ResNet50-IBN-a + GlobalAvgPool + BNNeck + Triplet + CE.

    - Backbone: ResNet-50-IBN-a with last_stride=1
    - Spatial Pool: Global Average Pooling (2048-d)
    - Temporal Pool: TemporalPool (attention/mean/max)
    - Head: BNNeckHead (pre-BN feature for triplet, post-BN for CE)
    """

    def __init__(self, cfg):
        super().__init__()
        self.num_classes = getattr(cfg, "num_classes", 0)
        self.chunk_size = getattr(cfg, "chunk_size", 64)
        pooling = getattr(cfg, "pooling_type", "attention")

        self.backbone = ResNet50_IBN_a(last_stride=1, with_nl=False, pretrained=True)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.temporal_pool = TemporalPool(2048, mode=pooling)
        self.head = BNNeckHead(2048, self.num_classes)

    def forward(self, x: torch.Tensor, targets: torch.Tensor = None):
        feat_map, B, T = _chunked_backbone_forward(self.backbone, x, self.chunk_size)
        feat_spatial = self.pool(feat_map).view(feat_map.size(0), 2048)

        if T > 1:
            feat_temporal = feat_spatial.view(T, B, 2048).transpose(0, 1)
            feat = self.temporal_pool(feat_temporal)
        else:
            feat = feat_spatial

        if not self.training:
            _, feat_ret, _ = self.head(feat)
            return feat_ret

        feat_tri, feat_ret, logits = self.head(feat)
        return feat_tri, logits

    def compute_loss(self, outputs, targets):
        feat_tri, logits = outputs
        loss_tri = triplet_loss(feat_tri, targets, margin=0.3, norm_feat=False, hard_mining=True)
        loss_cls = cross_entropy_loss(logits, targets, eps=0.1) if logits is not None else torch.tensor(0.0, device=feat_tri.device)
        total_loss = loss_tri + loss_cls
        return total_loss, loss_tri, loss_cls


# --- 2. AGW ---

class NativeAGW(nn.Module):
    """Native AGW (Attention Generalized Weighting):

    - Backbone: ResNet-50-IBN-a with Non-Local blocks in stage 2 & 3
    - Spatial Pool: GeneralizedMeanPooling (p=3.0)
    - Temporal Pool: TemporalPool
    - Head: BNNeckHead
    - Loss: Weighted Regularized Triplet Loss (soft-margin, margin=0.0) + CE
    """

    def __init__(self, cfg):
        super().__init__()
        self.num_classes = getattr(cfg, "num_classes", 0)
        self.chunk_size = getattr(cfg, "chunk_size", 64)
        pooling = getattr(cfg, "pooling_type", "attention")

        self.backbone = ResNet50_IBN_a(last_stride=1, with_nl=True, pretrained=True)
        self.pool = GeneralizedMeanPooling(norm=3.0)
        self.temporal_pool = TemporalPool(2048, mode=pooling)
        self.head = BNNeckHead(2048, self.num_classes)

    def forward(self, x: torch.Tensor, targets: torch.Tensor = None):
        feat_map, B, T = _chunked_backbone_forward(self.backbone, x, self.chunk_size)
        feat_spatial = self.pool(feat_map).view(feat_map.size(0), 2048)

        if T > 1:
            feat_temporal = feat_spatial.view(T, B, 2048).transpose(0, 1)
            feat = self.temporal_pool(feat_temporal)
        else:
            feat = feat_spatial

        if not self.training:
            _, feat_ret, _ = self.head(feat)
            return feat_ret

        feat_tri, feat_ret, logits = self.head(feat)
        return feat_tri, logits

    def compute_loss(self, outputs, targets):
        feat_tri, logits = outputs
        # AGW uses weighted example mining with soft-margin (margin=0.0)
        loss_tri = triplet_loss(feat_tri, targets, margin=0.0, norm_feat=False, hard_mining=False)
        loss_cls = cross_entropy_loss(logits, targets, eps=0.1) if logits is not None else torch.tensor(0.0, device=feat_tri.device)
        total_loss = loss_tri + loss_cls
        return total_loss, loss_tri, loss_cls


# --- 3. SBS ---

class NativeSBS(nn.Module):
    """Native SBS (Stronger Baseline):

    - Backbone: ResNet-50-IBN-a with Non-Local blocks in stage 2 & 3
    - Spatial Pool: GeneralizedMeanPoolingP (learnable p, init=3.0)
    - Temporal Pool: TemporalPool
    - Head: CircleSoftmaxHead (CircleSoftmax s=64, m=0.35)
    - Metric Loss: Soft-margin Triplet Loss (margin=0.0, batch-hard) on post-BN features
    """

    def __init__(self, cfg):
        super().__init__()
        self.num_classes = getattr(cfg, "num_classes", 0)
        self.chunk_size = getattr(cfg, "chunk_size", 64)
        pooling = getattr(cfg, "pooling_type", "attention")

        self.backbone = ResNet50_IBN_a(last_stride=1, with_nl=True, pretrained=True)
        self.pool = GeneralizedMeanPoolingP(norm=3.0)
        self.temporal_pool = TemporalPool(2048, mode=pooling)
        self.head = CircleSoftmaxHead(2048, self.num_classes, scale=64.0, margin=0.35)

    def forward(self, x: torch.Tensor, targets: torch.Tensor = None):
        feat_map, B, T = _chunked_backbone_forward(self.backbone, x, self.chunk_size)
        feat_spatial = self.pool(feat_map).view(feat_map.size(0), 2048)

        if T > 1:
            feat_temporal = feat_spatial.view(T, B, 2048).transpose(0, 1)
            feat = self.temporal_pool(feat_temporal)
        else:
            feat = feat_spatial

        if not self.training:
            return self.head(feat)

        feat_tri, feat_ret, cls_outputs = self.head(feat, targets)
        return feat_tri, cls_outputs

    def compute_loss(self, outputs, targets):
        feat_tri, cls_outputs = outputs
        # SBS uses batch-hard soft-margin triplet loss (margin=0.0) on post-BN features
        loss_tri = triplet_loss(feat_tri, targets, margin=0.0, norm_feat=False, hard_mining=True)
        loss_cls = cross_entropy_loss(cls_outputs, targets, eps=0.1) if cls_outputs is not None else torch.tensor(0.0, device=feat_tri.device)
        total_loss = loss_tri + loss_cls
        return total_loss, loss_tri, loss_cls


# --- 4. MGN ---

class NativeMGN(nn.Module):
    """Native MGN (Multiple Granularities Network):

    - Shared backbone: ResNet50-IBN-a through layer3[0]
    - 3 Branches:
        * b1: layer3[1:] + layer4 (stride 2) -> global pool
        * b2: layer3[1:] + layer4 (stride 1) -> global pool + 2 horizontal stripe pools
        * b3: layer3[1:] + layer4 (stride 1) -> global pool + 3 horizontal stripe pools
    - 8 TemporalPool modules (one per spatial feature)
    - 8 independent BNNeck heads
    - Eval feature: 16,384-dimensional concatenated normalized feature
    - Losses: 8 CE losses (scale 0.125) + 5 Triplet losses (scale 0.2, margin 0.3)
    """

    def __init__(self, cfg):
        super().__init__()
        self.num_classes = getattr(cfg, "num_classes", 0)
        self.chunk_size = getattr(cfg, "chunk_size", 64)
        pooling = getattr(cfg, "pooling_type", "attention")

        # Build full backbone to extract and copy weights
        full_backbone = ResNet50_IBN_a(last_stride=1, with_nl=False, pretrained=True)

        # 1. Shared backbone: conv1, bn1, relu, maxpool, layer1, layer2, layer3[0]
        self.shared_base = nn.Sequential(
            full_backbone.conv1,
            full_backbone.bn1,
            full_backbone.relu,
            full_backbone.maxpool,
            full_backbone.layer1,
            full_backbone.layer2,
            full_backbone.layer3[0]
        )

        # layer3[1:]
        res_conv4 = nn.Sequential(*full_backbone.layer3[1:])

        # layer4 with stride 2 (for b1)
        res_g_conv5 = ResNet50_IBN_a(last_stride=2, with_nl=False, pretrained=False).layer4
        res_g_conv5.load_state_dict(full_backbone.layer4.state_dict())

        # layer4 with stride 1 (for b2 & b3)
        res_p_conv5 = ResNet50_IBN_a(last_stride=1, with_nl=False, pretrained=False).layer4
        res_p_conv5.load_state_dict(full_backbone.layer4.state_dict())

        # Branch 1
        self.b1 = nn.Sequential(
            copy.deepcopy(res_conv4),
            copy.deepcopy(res_g_conv5)
        )
        self.b1_head = BNNeckHead(2048, self.num_classes)

        # Branch 2
        self.b2 = nn.Sequential(
            copy.deepcopy(res_conv4),
            copy.deepcopy(res_p_conv5)
        )
        self.b2_head = BNNeckHead(2048, self.num_classes)
        self.b21_head = BNNeckHead(2048, self.num_classes)
        self.b22_head = BNNeckHead(2048, self.num_classes)

        # Branch 3
        self.b3 = nn.Sequential(
            copy.deepcopy(res_conv4),
            copy.deepcopy(res_p_conv5)
        )
        self.b3_head = BNNeckHead(2048, self.num_classes)
        self.b31_head = BNNeckHead(2048, self.num_classes)
        self.b32_head = BNNeckHead(2048, self.num_classes)
        self.b33_head = BNNeckHead(2048, self.num_classes)

        # 8 Temporal Pools
        self.pools = nn.ModuleList([TemporalPool(2048, mode=pooling) for _ in range(8)])

    def _forward_branch_features(self, x_chunks):
        """Pass frame chunks through shared base and 3 branches."""
        shared_feats = [self.shared_base(c) for c in x_chunks]
        shared = torch.cat(shared_feats, dim=0)

        # Branch outputs
        b1_map = self.b1(shared)  # (N, 2048, H/32, W/32)
        b2_map = self.b2(shared)  # (N, 2048, H/16, W/16)
        b3_map = self.b3(shared)  # (N, 2048, H/16, W/16)

        # Spatial stripe splitting
        b1_spatial = F.adaptive_avg_pool2d(b1_map, (1, 1)).view(shared.size(0), 2048)

        b2_spatial = F.adaptive_avg_pool2d(b2_map, (1, 1)).view(shared.size(0), 2048)
        s21, s22 = torch.chunk(b2_map, 2, dim=2)
        b21_spatial = F.adaptive_avg_pool2d(s21, (1, 1)).view(shared.size(0), 2048)
        b22_spatial = F.adaptive_avg_pool2d(s22, (1, 1)).view(shared.size(0), 2048)

        b3_spatial = F.adaptive_avg_pool2d(b3_map, (1, 1)).view(shared.size(0), 2048)
        s31, s32, s33 = torch.chunk(b3_map, 3, dim=2)
        b31_spatial = F.adaptive_avg_pool2d(s31, (1, 1)).view(shared.size(0), 2048)
        b32_spatial = F.adaptive_avg_pool2d(s32, (1, 1)).view(shared.size(0), 2048)
        b33_spatial = F.adaptive_avg_pool2d(s33, (1, 1)).view(shared.size(0), 2048)

        return [
            b1_spatial, b2_spatial, b21_spatial, b22_spatial,
            b3_spatial, b31_spatial, b32_spatial, b33_spatial
        ]

    def forward(self, x: torch.Tensor, targets: torch.Tensor = None):
        if x.dim() == 5:
            B, T, C, H, W = x.shape
            frames = x.transpose(0, 1).reshape(T * B, C, H, W)
            chunks = torch.split(frames, self.chunk_size, dim=0)
        else:
            B, T = x.size(0), 1
            chunks = [x]

        spatial_8 = self._forward_branch_features(chunks)

        # Temporal aggregation per branch
        features_8 = []
        for i, feat in enumerate(spatial_8):
            if T > 1:
                feat_t = feat.view(T, B, 2048).transpose(0, 1)
                features_8.append(self.pools[i](feat_t))
            else:
                features_8.append(feat)

        heads = [
            self.b1_head, self.b2_head, self.b21_head, self.b22_head,
            self.b3_head, self.b31_head, self.b32_head, self.b33_head
        ]

        if not self.training:
            # Concat all 8 post-BN normalized features: 8 * 2048 = 16384 dimensions
            ret_feats = [h(f)[1] for h, f in zip(heads, features_8)]
            return F.normalize(torch.cat(ret_feats, dim=1), dim=-1)

        # Training pass
        tri_feats = []
        ret_feats = []
        logits = []
        for h, f in zip(heads, features_8):
            tri, ret, logit = h(f)
            tri_feats.append(tri)
            ret_feats.append(ret)
            logits.append(logit)

        # MGN groups triplet features: b1, b2, b3, cat(b21, b22), cat(b31, b32, b33)
        b22_combined = torch.cat([tri_feats[2], tri_feats[3]], dim=1)
        b33_combined = torch.cat([tri_feats[5], tri_feats[6], tri_feats[7]], dim=1)
        mgn_triplet_feats = [
            tri_feats[0],  # b1
            tri_feats[1],  # b2
            tri_feats[4],  # b3
            b22_combined,  # b21 + b22
            b33_combined   # b31 + b32 + b33
        ]

        return mgn_triplet_feats, logits

    def compute_loss(self, outputs, targets):
        mgn_triplet_feats, logits = outputs

        # 8 classification losses (scale 0.125 each, sum = 1.0)
        loss_cls = sum(
            cross_entropy_loss(l, targets, eps=0.1) for l in logits if l is not None
        ) * 0.125

        # 5 triplet losses (scale 0.2 each, margin 0.3, sum = 1.0)
        loss_tri = sum(
            triplet_loss(f, targets, margin=0.3, norm_feat=False, hard_mining=True)
            for f in mgn_triplet_feats
        ) * 0.2

        total_loss = loss_tri + loss_cls
        return total_loss, loss_tri, loss_cls
