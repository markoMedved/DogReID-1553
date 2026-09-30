import os
import sys
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F

# Ensure OpenAnimals is importable
PROJECT_ROOT = Path(__file__).resolve().parent.parent
OPENANIMALS_DIR = PROJECT_ROOT / "OpenAnimals"
if str(OPENANIMALS_DIR) not in sys.path:
    sys.path.insert(0, str(OPENANIMALS_DIR))

from openanimals.config import get_cfg
from openanimals.modeling import build_model as build_oa_model
from .reid_model import TemporalPool


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
            self.embed_dim = oa_cfg.MODEL.BACKBONE.FEAT_DIM
            self.temporal_pool = TemporalPool(self.embed_dim, mode=pooling_type)

    def _forward_baseline(self, x, targets=None):
        if x.dim() == 5:
            B, T, C, H, W = x.shape
            frames = x.view(B * T, C, H, W)

            # Chunked backbone forward pass to control activation memory
            chunks = torch.split(frames, self.chunk_size, dim=0)
            chunk_feats = []
            for chunk in chunks:
                f_map = self.oa_model.backbone(chunk)
                f_pool = self.oa_model.heads.pool_layer(f_map).flatten(1)
                chunk_feats.append(f_pool)

            frame_feats = torch.cat(chunk_feats, dim=0).view(B, T, -1)
            video_feat = self.temporal_pool(frame_feats)  # (B, 2048)
        else:
            f_map = self.oa_model.backbone(x)
            video_feat = self.oa_model.heads.pool_layer(f_map).flatten(1)

        pool_feat = video_feat.view(video_feat.size(0), -1, 1, 1)
        neck_feat = self.oa_model.heads.bottleneck(pool_feat)[..., 0, 0]

        # Evaluation mode: extract normalized features after BNNeck
        if not self.training:
            return F.normalize(neck_feat, p=2, dim=-1)

        # Training mode: compute logits
        if self.oa_model.heads.cls_layer.__class__.__name__ == 'Linear':
            logits = F.linear(neck_feat, self.oa_model.heads.weight)
            cls_outputs = logits
        else:
            norm_neck = F.normalize(neck_feat, p=2, dim=-1)
            norm_weight = F.normalize(self.oa_model.heads.weight, p=2, dim=-1)
            logits = F.linear(norm_neck, norm_weight)
            if targets is not None:
                cls_outputs = self.oa_model.heads.cls_layer(logits.clone(), targets)
            else:
                scale = getattr(self.oa_model.heads.cls_layer, 's', 1.0)
                cls_outputs = logits.mul(scale)

        feat = video_feat if getattr(self.oa_model.heads, 'neck_feat', 'after') == 'before' else neck_feat
        embeddings = F.normalize(feat, p=2, dim=-1)
        return embeddings, cls_outputs

    def _forward_mgn(self, x, targets=None):
        if x.dim() == 5:
            B, T, C, H, W = x.shape
            frames = x.view(B * T, C, H, W)
        else:
            B = x.size(0)
            T = 1
            frames = x

        chunks = torch.split(frames, self.chunk_size, dim=0)
        chunk_pooled = [[] for _ in range(8)]
        for chunk in chunks:
            feat_chunk = self.oa_model.backbone(chunk)
            b1_chunk = self.oa_model.b1(feat_chunk)
            b2_chunk = self.oa_model.b2(feat_chunk)
            b3_chunk = self.oa_model.b3(feat_chunk)
            b21_c, b22_c = torch.chunk(b2_chunk, 2, dim=2)
            b31_c, b32_c, b33_c = torch.chunk(b3_chunk, 3, dim=2)

            maps = [b1_chunk, b2_chunk, b21_c, b22_c, b3_chunk, b31_c, b32_c, b33_c]
            for idx in range(8):
                head = getattr(self.oa_model, f"{['b1', 'b2', 'b21', 'b22', 'b3', 'b31', 'b32', 'b33'][idx]}_head")
                p = head.pool_layer(maps[idx]).flatten(1)
                chunk_pooled[idx].append(p)

        branches = [
            (self.oa_model.b1_head, self.temporal_pools[0]),
            (self.oa_model.b2_head, self.temporal_pools[1]),
            (self.oa_model.b21_head, self.temporal_pools[2]),
            (self.oa_model.b22_head, self.temporal_pools[3]),
            (self.oa_model.b3_head, self.temporal_pools[4]),
            (self.oa_model.b31_head, self.temporal_pools[5]),
            (self.oa_model.b32_head, self.temporal_pools[6]),
            (self.oa_model.b33_head, self.temporal_pools[7]),
        ]

        pooled_feats = []
        for idx, (head, pool) in enumerate(branches):
            all_chunks = torch.cat(chunk_pooled[idx], dim=0)
            v = pool(all_chunks.view(B, T, 2048)) if T > 1 else all_chunks
            pooled_feats.append(v)

        necks = []
        for (head, pool), v in zip(branches, pooled_feats):
            neck = head.bottleneck(v.view(B, 2048, 1, 1))[..., 0, 0]
            necks.append(neck)

        pre_bn_feat = torch.cat(pooled_feats, dim=1)
        eval_feat = torch.cat(necks, dim=1)

        # Evaluation mode: concatenate all 8 neck features (16,384-D)
        if not self.training:
            return F.normalize(eval_feat, p=2, dim=-1)

        # Training mode: compute logits for all 8 branches
        logits_list = []
        for (head, pool), neck in zip(branches, necks):
            if head.cls_layer.__class__.__name__ == 'Linear':
                logit = F.linear(neck, head.weight)
                cls_out = logit
            else:
                norm_neck = F.normalize(neck, p=2, dim=-1)
                norm_weight = F.normalize(head.weight, p=2, dim=-1)
                logit = F.linear(norm_neck, norm_weight)
                if targets is not None:
                    cls_out = head.cls_layer(logit.clone(), targets)
                else:
                    scale = getattr(head.cls_layer, 's', 1.0)
                    cls_out = logit.mul(scale)
            logits_list.append(cls_out)

        feat = pre_bn_feat if getattr(self.oa_model.b1_head, 'neck_feat', 'before') == 'before' else eval_feat
        embeddings = F.normalize(feat, p=2, dim=-1)
        return embeddings, logits_list

    def forward(self, x, targets=None):
        if self.is_mgn:
            return self._forward_mgn(x, targets=targets)
        return self._forward_baseline(x, targets=targets)

