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
from openanimals.modeling.losses.cross_entroy_loss import cross_entropy_loss
from openanimals.modeling.losses.triplet_loss import triplet_loss
from openanimals.modeling.losses.circle_loss import pairwise_circleloss
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
        self.is_mgn = "mgn" in clean_name
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

        # Evaluation mode: extract normalized features according to paper
        if not self.training:
            if getattr(self.oa_model.heads, 'neck_feat', 'after') == 'before':
                return F.normalize(video_feat, p=2, dim=-1)
            return F.normalize(neck_feat, p=2, dim=-1)

        # Training mode: compute OpenAnimals native heads and losses
        if self.oa_model.heads.cls_layer.__class__.__name__ == 'Linear':
            logits = F.linear(neck_feat, self.oa_model.heads.weight)
        else:
            logits = F.linear(F.normalize(neck_feat), F.normalize(self.oa_model.heads.weight))

        cls_outputs = self.oa_model.heads.cls_layer(logits.clone(), targets)
        feat = video_feat if getattr(self.oa_model.heads, 'neck_feat', 'after') == 'before' else neck_feat

        # Compute native OpenAnimals losses
        loss_dict = {}
        loss_names = self.oa_cfg.MODEL.LOSSES.NAME
        if 'CrossEntropyLoss' in loss_names and targets is not None:
            loss_dict['loss_cls'] = cross_entropy_loss(
                cls_outputs,
                targets,
                self.oa_cfg.MODEL.LOSSES.CE.EPSILON,
                self.oa_cfg.MODEL.LOSSES.CE.ALPHA
            ) * self.oa_cfg.MODEL.LOSSES.CE.SCALE

        if 'TripletLoss' in loss_names and targets is not None:
            loss_dict['loss_triplet'] = triplet_loss(
                feat,
                targets,
                self.oa_cfg.MODEL.LOSSES.TRI.MARGIN,
                self.oa_cfg.MODEL.LOSSES.TRI.NORM_FEAT,
                self.oa_cfg.MODEL.LOSSES.TRI.HARD_MINING
            ) * self.oa_cfg.MODEL.LOSSES.TRI.SCALE

        if 'CircleLoss' in loss_names and targets is not None:
            loss_dict['loss_circle'] = pairwise_circleloss(
                feat,
                targets,
                self.oa_cfg.MODEL.LOSSES.CIRCLE.MARGIN,
                self.oa_cfg.MODEL.LOSSES.CIRCLE.GAMMA
            ) * self.oa_cfg.MODEL.LOSSES.CIRCLE.SCALE

        total_loss = sum(loss_dict.values()) if loss_dict else torch.tensor(0.0, device=video_feat.device)
        return feat, cls_outputs, total_loss, loss_dict

    def _forward_mgn(self, x, targets=None):
        if x.dim() == 5:
            B, T, C, H, W = x.shape
            frames = x.view(B * T, C, H, W)
        else:
            B = x.size(0)
            T = 1
            frames = x

        feat = self.oa_model.backbone(frames)
        b1 = self.oa_model.b1(feat)
        b2 = self.oa_model.b2(feat)
        b3 = self.oa_model.b3(feat)
        b21, b22 = torch.chunk(b2, 2, dim=2)
        b31, b32, b33 = torch.chunk(b3, 3, dim=2)

        branches = [
            (b1, self.oa_model.b1_head, self.temporal_pools[0]),
            (b2, self.oa_model.b2_head, self.temporal_pools[1]),
            (b21, self.oa_model.b21_head, self.temporal_pools[2]),
            (b22, self.oa_model.b22_head, self.temporal_pools[3]),
            (b3, self.oa_model.b3_head, self.temporal_pools[4]),
            (b31, self.oa_model.b31_head, self.temporal_pools[5]),
            (b32, self.oa_model.b32_head, self.temporal_pools[6]),
            (b33, self.oa_model.b33_head, self.temporal_pools[7]),
        ]

        pooled_feats = []
        for idx, (f_map, head, pool) in enumerate(branches):
            p = head.pool_layer(f_map).flatten(1)
            v = pool(p.view(B, T, 2048)) if T > 1 else p
            pooled_feats.append(v)

        # Evaluation mode: concatenate all 8 neck features (16,384-D)
        if not self.training:
            necks = []
            for (f_map, head, pool), v in zip(branches, pooled_feats):
                neck = head.bottleneck(v.view(B, 2048, 1, 1))[..., 0, 0]
                necks.append(neck)
            eval_feat = torch.cat(necks, dim=1)
            return F.normalize(eval_feat, p=2, dim=-1)

        # Training mode: run all 8 OpenAnimals heads
        branch_outputs = []
        for (f_map, head, pool), v in zip(branches, pooled_feats):
            out = head(v.view(B, 2048, 1, 1), targets)
            branch_outputs.append(out)

        # Exact OpenAnimals MGN multi-task loss computation
        loss_dict = {}
        if targets is not None:
            # 8-branch cross-entropy with 0.125 weight
            ce_losses = [
                cross_entropy_loss(
                    out['cls_outputs'],
                    targets,
                    self.oa_cfg.MODEL.LOSSES.CE.EPSILON,
                    self.oa_cfg.MODEL.LOSSES.CE.ALPHA
                ) * 0.125
                for out in branch_outputs
            ]
            loss_dict['loss_cls'] = sum(ce_losses) * self.oa_cfg.MODEL.LOSSES.CE.SCALE

            # 5-branch triplet loss on global and concatenated part stripes
            b22_pool = torch.cat([branch_outputs[2]['features'], branch_outputs[3]['features']], dim=1)
            b33_pool = torch.cat([branch_outputs[5]['features'], branch_outputs[6]['features'], branch_outputs[7]['features']], dim=1)
            tri_cfg = self.oa_cfg.MODEL.LOSSES.TRI
            t1 = triplet_loss(branch_outputs[0]['features'], targets, tri_cfg.MARGIN, tri_cfg.NORM_FEAT, tri_cfg.HARD_MINING)
            t2 = triplet_loss(branch_outputs[1]['features'], targets, tri_cfg.MARGIN, tri_cfg.NORM_FEAT, tri_cfg.HARD_MINING)
            t22 = triplet_loss(b22_pool, targets, tri_cfg.MARGIN, tri_cfg.NORM_FEAT, tri_cfg.HARD_MINING)
            t3 = triplet_loss(branch_outputs[4]['features'], targets, tri_cfg.MARGIN, tri_cfg.NORM_FEAT, tri_cfg.HARD_MINING)
            t33 = triplet_loss(b33_pool, targets, tri_cfg.MARGIN, tri_cfg.NORM_FEAT, tri_cfg.HARD_MINING)
            loss_dict['loss_triplet'] = (t1 + t2 + t22 + t3 + t33) * tri_cfg.SCALE

        total_loss = sum(loss_dict.values()) if loss_dict else torch.tensor(0.0, device=x.device)
        return branch_outputs[0]['features'], branch_outputs[0]['cls_outputs'], total_loss, loss_dict

    def forward(self, x, targets=None):
        if self.is_mgn:
            return self._forward_mgn(x, targets=targets)
        return self._forward_baseline(x, targets=targets)
