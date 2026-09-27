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
        oa_cfg.MODEL.HEADS.NUM_CLASSES = num_classes if num_classes > 0 else 776

        self.oa_model = build_oa_model(oa_cfg)
        self.is_mgn = "mgn" in clean_name
        self.chunk_size = chunk_size
        self.num_classes = num_classes
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

    def _forward_baseline(self, x):
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

        # Pass through OpenAnimals BNNeck
        neck_feat = self.oa_model.heads.bottleneck(video_feat.view(video_feat.size(0), -1, 1, 1))[..., 0, 0]

        if self.training and self.num_classes > 0:
            logits = F.linear(neck_feat, self.oa_model.heads.weight)
            embeddings = F.normalize(video_feat, dim=-1)
            return embeddings, logits
        else:
            return F.normalize(neck_feat, p=2, dim=-1)

    def _forward_mgn(self, x):
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

        necks = []
        logits_list = []
        global_v = None

        for idx, (f_map, head, pool) in enumerate(branches):
            p = head.pool_layer(f_map).flatten(1)
            if T > 1:
                v = pool(p.view(B, T, 2048))
            else:
                v = p
            if idx == 0:
                global_v = v

            neck = head.bottleneck(v.view(B, 2048, 1, 1))[..., 0, 0]
            logit = F.linear(neck, head.weight)
            necks.append(neck)
            logits_list.append(logit)

        if self.training and self.num_classes > 0:
            embeddings = F.normalize(global_v, dim=-1)
            cls_score = torch.stack(logits_list, dim=0).mean(0)
            return embeddings, cls_score
        else:
            eval_feat = torch.cat(necks, dim=1)
            return F.normalize(eval_feat, p=2, dim=-1)

    def forward(self, x):
        if self.is_mgn:
            return self._forward_mgn(x)
        return self._forward_baseline(x)
