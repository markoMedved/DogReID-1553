import os
import sys
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from contextlib import contextmanager, nullcontext

# Ensure OpenAnimals is importable
PROJECT_ROOT = Path(__file__).resolve().parent.parent
OPENANIMALS_DIR = PROJECT_ROOT / "OpenAnimals"
if str(OPENANIMALS_DIR) not in sys.path:
    sys.path.insert(0, str(OPENANIMALS_DIR))

from openanimals.config import get_cfg
from openanimals.modeling import build_model as build_oa_model
from openanimals.modeling.backbones.build import BACKBONE_REGISTRY
from .reid_model import DINOv2Adapter, TemporalPool


class DINOv2OpenAnimalsBackbone(nn.Module):
    def __init__(self, variant="vitb14_reg"):
        super().__init__()
        self.adapter = DINOv2Adapter(variant)

    def forward(self, x):
        feat = self.adapter.pooled(x)
        return feat.view(feat.size(0), feat.size(1), 1, 1)


if "build_dinov2_backbone" not in getattr(BACKBONE_REGISTRY, "_obj_map", {}):
    @BACKBONE_REGISTRY.register()
    def build_dinov2_backbone(cfg):
        variant = getattr(cfg.MODEL.BACKBONE, "VARIANT", "vitb14_reg")
        return DINOv2OpenAnimalsBackbone(variant)


class OpenAnimalsVideoModel(nn.Module):
    """
    Adapts OpenAnimals architectures (BoT, AGW, SBS, MGN, ARBase) for video Re-ID
    with temporal pooling (attention, mean, max) across video clip frames.
    Uses OpenAnimals native loss formulations, heads, and backbones.
    """

    MODEL_CONFIGS = {
        "bot": "OpenAnimals/configs/DogReID/bot.yml",
        "oa_bot": "OpenAnimals/configs/DogReID/bot.yml",
        "openanimals_bot": "OpenAnimals/configs/DogReID/bot.yml",

        "oa_dinov2": "OpenAnimals/configs/DogReID/bot.yml",
        "oa_dinov2_bot": "OpenAnimals/configs/DogReID/bot.yml",
        "openanimals_dinov2_bot": "OpenAnimals/configs/DogReID/bot.yml",

        "agw": "OpenAnimals/configs/DogReID/agw.yml",
        "oa_agw": "OpenAnimals/configs/DogReID/agw.yml",
        "openanimals_agw": "OpenAnimals/configs/DogReID/agw.yml",

        "sbs": "OpenAnimals/configs/DogReID/sbs.yml",
        "oa_sbs": "OpenAnimals/configs/DogReID/sbs.yml",
        "openanimals_sbs": "OpenAnimals/configs/DogReID/sbs.yml",

        "mgn": "OpenAnimals/configs/DogReID/mgn.yml",
        "oa_mgn": "OpenAnimals/configs/DogReID/mgn.yml",
        "openanimals_mgn": "OpenAnimals/configs/DogReID/mgn.yml",

        "arbase": "OpenAnimals/configs/DogReID/arbase.yml",
        "oa_arbase": "OpenAnimals/configs/DogReID/arbase.yml",
        "openanimals_arbase": "OpenAnimals/configs/DogReID/arbase.yml",

        "arbase_mb": "OpenAnimals/configs/DogReID/mgn.yml",
        "oa_arbase_mb": "OpenAnimals/configs/DogReID/mgn.yml",
        "arbase_mgn": "OpenAnimals/configs/DogReID/mgn.yml",
    }

    def __init__(self, model_name: str, pooling_type: str = "attention", num_classes: int = 0, chunk_size: int = 32):
        super().__init__()
        clean_name = model_name.lower().replace("-", "_")
        if clean_name not in self.MODEL_CONFIGS:
            raise ValueError(f"Unknown OpenAnimals model: {model_name}. Available: {list(self.MODEL_CONFIGS.keys())}")

        cfg_rel_path = self.MODEL_CONFIGS[clean_name]
        cfg_abs_path = str(PROJECT_ROOT / cfg_rel_path)

        # Build OpenAnimals configuration
        oa_cfg = get_cfg()
        oa_cfg.merge_from_file(cfg_abs_path)
        oa_cfg = oa_cfg.clone()
        oa_cfg.defrost()
        oa_cfg.MODEL.DEVICE = "cpu"  # Initialized on CPU; moved to GPU via .to(device)
        oa_cfg.MODEL.BACKBONE.PRETRAIN = True
        oa_cfg.MODEL.HEADS.NUM_CLASSES = num_classes if num_classes > 0 else 1553

        is_dinov2 = "dinov2" in clean_name
        if is_dinov2:
            oa_cfg.MODEL.BACKBONE.NAME = "build_dinov2_backbone"
            oa_cfg.MODEL.BACKBONE.FEAT_DIM = 768
            oa_cfg.MODEL.HEADS.IN_FEAT = 768
            oa_cfg.INPUT.SIZE_TRAIN = [224, 224]
            oa_cfg.INPUT.SIZE_TEST = [224, 224]

        self.oa_cfg = oa_cfg
        self.is_openanimals = True
        self.oa_model = build_oa_model(oa_cfg)
        self.is_mgn = "mgn" in clean_name or "arbase_mb" in clean_name or "arbase_mgn" in clean_name
        self.chunk_size = chunk_size
        self.num_classes = oa_cfg.MODEL.HEADS.NUM_CLASSES
        self.pooling_type = pooling_type

        if self.is_mgn:
            # MGN has 8 branches: b1, b2, b21, b22, b3, b31, b32, b33
            self.temporal_pools = nn.ModuleList(
                [TemporalPool(2048, mode=pooling_type) for _ in range(8)]
            )
            self.embed_dim = 2048 * 8
        else:
            self.embed_dim = 768 if is_dinov2 else oa_cfg.MODEL.BACKBONE.FEAT_DIM
            self.temporal_pool = TemporalPool(self.embed_dim, mode=pooling_type)

    # ------------------------------------------------------------------
    # Backbone helpers
    # ------------------------------------------------------------------
    def _to_time_major(self, x):
        """(B, T, C, H, W) -> (T*B, C, H, W), ordered frame-index first.

        With chunk_size == B, every backbone chunk then contains frame t of
        every clip in the batch, so BatchNorm statistics are computed over B
        different clips (P identities) -- the same composition as an
        OpenAnimals image batch -- instead of over T near-identical frames
        of a single clip.
        """
        B, T, C, H, W = x.shape
        return x.transpose(0, 1).reshape(T * B, C, H, W), B, T

    @staticmethod
    def _from_time_major(feats, B, T):
        """(T*B, D) -> (B, T, D)."""
        return feats.view(T, B, -1).transpose(0, 1)

    def _run_chunks(self, frames, fn):
        """Run fn over frame chunks. During training, uses activation
        checkpointing; BN running stats are frozen during the backward
        recompute so they are updated exactly once per step."""
        outs = []
        use_ckpt = self.training and torch.is_grad_enabled()
        for chunk in torch.split(frames, self.chunk_size, dim=0):
            if use_ckpt:
                out = checkpoint(
                    fn, chunk, use_reentrant=False,
                    context_fn=lambda: (nullcontext(), _frozen_bn_stats(self.oa_model)),
                )
            else:
                out = fn(chunk)
            outs.append(out)
        return outs

    # ------------------------------------------------------------------
    # Single-branch models: BoT / AGW / SBS / ARBase
    # ------------------------------------------------------------------
    def _forward_baseline(self, x, targets=None):
        heads = self.oa_model.heads

        def _eval_chunk(c):
            return heads.pool_layer(self.oa_model.backbone(c)).flatten(1)

        if x.dim() == 5:
            frames, B, T = self._to_time_major(x)
            frame_feats = torch.cat(self._run_chunks(frames, _eval_chunk), dim=0)
            video_feat = self.temporal_pool(self._from_time_major(frame_feats, B, T))  # (B, 2048)
        else:
            video_feat = _eval_chunk(x)

        neck_feat = heads.bottleneck(video_feat.view(video_feat.size(0), -1, 1, 1))[..., 0, 0]

        # Evaluation: post-BNNeck features (OpenAnimals EmbeddingHead eval output)
        if not self.training:
            return F.normalize(neck_feat, p=2, dim=-1)

        # Training logits (mirrors EmbeddingHead.forward)
        if heads.cls_layer.__class__.__name__ == 'Linear':
            logits = F.linear(neck_feat, heads.weight)
        else:
            logits = F.linear(F.normalize(neck_feat), F.normalize(heads.weight))
        if targets is not None:
            cls_outputs = heads.cls_layer(logits.clone(), targets)
        else:
            cls_outputs = logits.mul(getattr(heads.cls_layer, 's', 1.0))

        # Triplet features: raw (unnormalised), selected by NECK_FEAT,
        # because OpenAnimals uses NORM_FEAT=False (Euclidean on raw features).
        feat = video_feat if heads.neck_feat == 'before' else neck_feat
        return feat, cls_outputs

    # ------------------------------------------------------------------
    # MGN (8 branches)
    # ------------------------------------------------------------------
    def _forward_mgn(self, x, targets=None):
        m = self.oa_model
        head_names = ["b1", "b2", "b21", "b22", "b3", "b31", "b32", "b33"]
        heads = [getattr(m, f"{n}_head") for n in head_names]

        def _eval_mgn_chunk(c):
            feat = m.backbone(c)
            b1 = m.b1(feat)
            b2 = m.b2(feat)
            b3 = m.b3(feat)
            b21, b22 = torch.chunk(b2, 2, dim=2)
            b31, b32, b33 = torch.chunk(b3, 3, dim=2)
            maps = [b1, b2, b21, b22, b3, b31, b32, b33]
            return tuple(h.pool_layer(fm).flatten(1) for h, fm in zip(heads, maps))

        if x.dim() == 5:
            frames, B, T = self._to_time_major(x)
        else:
            frames, B, T = x, x.size(0), 1

        chunk_outs = self._run_chunks(frames, _eval_mgn_chunk)

        pooled = {}
        for idx, name in enumerate(head_names):
            f = torch.cat([co[idx] for co in chunk_outs], dim=0)
            if T > 1:
                f = self.temporal_pools[idx](self._from_time_major(f, B, T))
            pooled[name] = f

        necks = {
            name: h.bottleneck(pooled[name].view(B, -1, 1, 1))[..., 0, 0]
            for name, h in zip(head_names, heads)
        }

        # Evaluation: concatenation of all 8 post-BN features (OpenAnimals order)
        if not self.training:
            eval_order = ["b1", "b2", "b3", "b21", "b22", "b31", "b32", "b33"]
            return F.normalize(torch.cat([necks[n] for n in eval_order], dim=1), p=2, dim=-1)

        # Training: CE on all 8 branches (averaged in the trainer == 0.125 each)
        logits_list = []
        for name, h in zip(head_names, heads):
            neck = necks[name]
            if h.cls_layer.__class__.__name__ == 'Linear':
                logit = F.linear(neck, h.weight)
            else:
                logit = F.linear(F.normalize(neck), F.normalize(h.weight))
            if targets is not None:
                logits_list.append(h.cls_layer(logit.clone(), targets))
            else:
                logits_list.append(logit.mul(getattr(h.cls_layer, 's', 1.0)))

        # Triplet on 5 branches separately (OpenAnimals MGN.losses: b1, b2, b3,
        # b22, b33, each x0.2 -> averaged in the trainer), raw features.
        src = pooled if m.b1_head.neck_feat == 'before' else necks
        tri_feats = [src[n] for n in ["b1", "b2", "b3", "b22", "b33"]]
        return tri_feats, logits_list

    def forward(self, x, targets=None):
        if self.is_mgn:
            return self._forward_mgn(x, targets=targets)
        return self._forward_baseline(x, targets=targets)


@contextmanager
def _frozen_bn_stats(module):
    """Temporarily set BN momentum to 0 so the checkpoint recompute pass
    does not update running_mean/var a second time."""
    saved = []
    for mod in module.modules():
        if isinstance(mod, nn.modules.batchnorm._BatchNorm) and mod.training and mod.track_running_stats:
            saved.append((mod, mod.momentum))
            mod.momentum = 0.0
    try:
        yield
    finally:
        for mod, mom in saved:
            mod.momentum = mom
