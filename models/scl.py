import importlib.util
import sys
from pathlib import Path
import torch
import torch.nn as nn

from .reid_model import TemporalPool

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SCL_ROOT = PROJECT_ROOT / "ECCV-26-SCL"


def _load_scl_components():
    """Import ECCV-26-SCL config and model components without namespace collisions."""
    if str(SCL_ROOT) not in sys.path:
        sys.path.insert(0, str(SCL_ROOT))

    # Temporarily isolate 'model' / 'config' module names if needed
    saved_modules = {}
    for mod_name in ("config", "config.defaults", "model", "model.vit", "model.vit.common", "model.vit.vit_pytorch", "model.modules", "model.modules.freq_norm_vit", "model.make_model"):
        if mod_name in sys.modules:
            saved_modules[mod_name] = sys.modules.pop(mod_name)

    try:
        from config.defaults import _C as scl_default_cfg
        from model.vit.vit_pytorch import vit_base_patch16_224_TransReID
        from model.modules.freq_norm_vit import SCNorm1D, FGSCNorm2D
        from model.make_model import weights_init_kaiming, weights_init_classifier
        return scl_default_cfg, vit_base_patch16_224_TransReID, SCNorm1D, FGSCNorm2D, weights_init_kaiming, weights_init_classifier
    finally:
        # Restore project modules so our top-level packages remain untouched
        for k, v in saved_modules.items():
            sys.modules[k] = v


_SCL_CFG_PROTO, _vit_base_patch16_224_TransReID, _SCNorm1D, _FGSCNorm2D, _weights_init_kaiming, _weights_init_classifier = _load_scl_components()

# Ensure FFT operations inside FGSCNorm2D and SCNorm1D safely run in float32 under AMP
_orig_fgsc_forward = _FGSCNorm2D.forward
def _safe_fgsc_forward(self, x, fg_mask=None):
    orig_dtype = x.dtype
    out = _orig_fgsc_forward(self, x.float(), fg_mask=fg_mask.float() if fg_mask is not None else None)
    return out.to(orig_dtype)
_FGSCNorm2D.forward = _safe_fgsc_forward

_orig_sc1d_forward = _SCNorm1D.forward
def _safe_sc1d_forward(self, x):
    orig_dtype = x.dtype
    out = _orig_sc1d_forward(self, x.float())
    return out.to(orig_dtype)
_SCNorm1D.forward = _safe_sc1d_forward


def build_scl_cfg(cfg):
    """Build the exact ECCV-26-SCL YACS config from transreid_nfc_scfg_w10k29_to_full12.yml."""
    scl_cfg = _SCL_CFG_PROTO.clone()
    yml_path = SCL_ROOT / "configs" / "transreid_nfc_scfg_w10k29_to_full12.yml"
    if yml_path.exists():
        scl_cfg.merge_from_file(str(yml_path))

    scl_cfg.defrost()
    scl_cfg.MODEL.DIST_TRAIN = False
    pretrain_path = SCL_ROOT / "pretrained" / "jx_vit_base_p16_224-80ecf9dd.pth"
    scl_cfg.MODEL.PRETRAIN_PATH = str(pretrain_path)

    img_size = tuple(getattr(cfg, "img_size", (256, 256)))
    scl_cfg.INPUT.SIZE_TRAIN = list(img_size)
    scl_cfg.INPUT.SIZE_TEST = list(img_size)

    if getattr(cfg, "epochs", None) is not None:
        scl_cfg.SOLVER.MAX_EPOCHS = int(cfg.epochs)
    if getattr(cfg, "lr", None) is not None:
        scl_cfg.SOLVER.BASE_LR = float(cfg.lr)
    if getattr(cfg, "margin", None) is not None:
        scl_cfg.SOLVER.MARGIN = float(cfg.margin)
    if getattr(cfg, "weight_decay", None) is not None:
        scl_cfg.SOLVER.WEIGHT_DECAY = float(cfg.weight_decay)
        scl_cfg.SOLVER.WEIGHT_DECAY_BIAS = float(cfg.weight_decay)

    # DogReID-1553 is a single-species dataset (dogs), so use the standard reciprocal
    # mutual-neighbor NFC branch (find_mutual_neighbors with K1=8, K2=8)
    scl_cfg.CHANGE.METHODS.NFC_CROSS_SPECIES = False
    scl_cfg.freeze()
    return scl_cfg


class SCLVideoModel(nn.Module):
    """
    Video adaptation of ECCV-26-SCL (MetaN: TransReID ViT-B/16 + SCFG + SCNorm1D + NFC).
    Encodes each video frame with the SCFG-equipped TransReID ViT-B/16 backbone,
    aggregates frame features across time via TemporalPool ('mean' or 'attention'),
    and applies SCNorm1D + BNNeck + Classifier.
    """

    def __init__(self, cfg):
        super().__init__()
        self.is_scl = True
        self.scl_cfg = build_scl_cfg(cfg)
        self.pooling_type = str(getattr(cfg, "pooling_type", "mean") or "mean").lower()
        if self.pooling_type == "attn":
            self.pooling_type = "attention"
        self.chunk_size = int(getattr(cfg, "chunk_size", 64) or 64)
        self.num_classes = int(getattr(cfg, "num_classes", 0) or 0)
        self.neck_feat = self.scl_cfg.TEST.NECK_FEAT

        # 1. TransReID ViT-B/16 with SCFG at positions [0, 4, 8]
        self.image_encoder = _vit_base_patch16_224_TransReID(self.scl_cfg)
        self.in_planes = getattr(self.image_encoder, "embed_dim", 768)

        pretrain_path = Path(self.scl_cfg.MODEL.PRETRAIN_PATH)
        if pretrain_path.exists():
            self.image_encoder.load_param(str(pretrain_path))
            print(f"[SCL] Loaded ImageNet pretrained ViT weights from {pretrain_path}")
        else:
            print(f"[SCL] WARNING: Pretrained ViT file not found at {pretrain_path}")

        # 2. Temporal Pooling across frames ('mean' or 'attention')
        self.temporal_pool = TemporalPool(self.in_planes, mode=self.pooling_type)

        # 3. SCNorm1D before BNNeck
        self.scnorm_before_bn = None
        if self.scl_cfg.CHANGE.METHODS.SCNORM_BEFORE_BNNECK:
            self.scnorm_before_bn = _SCNorm1D(
                self.in_planes,
                base=self.scl_cfg.CHANGE.METHODS.SCNORM_BNNECK_BASE,
                Ts=1e-1,
            )

        # 4. BNNeck & Classifier
        self.bottleneck = nn.BatchNorm1d(self.in_planes)
        self.bottleneck.bias.requires_grad_(False)
        self.bottleneck.apply(_weights_init_kaiming)

        self.classifier = nn.Linear(self.in_planes, max(1, self.num_classes), bias=False)
        self.classifier.apply(_weights_init_classifier)

    def set_freq_norm_epoch(self, epoch: int):
        for module in self.modules():
            if hasattr(module, "set_epoch") and callable(module.set_epoch):
                module.set_epoch(epoch)

    def _encode_chunk(self, chunk: torch.Tensor) -> torch.Tensor:
        return self.image_encoder(forward_type="main", x=chunk)[:, 0]

    def _encode_frames(self, frames: torch.Tensor) -> torch.Tensor:
        chunk_sz = min(self.chunk_size, 16)
        if frames.size(0) <= chunk_sz:
            return self._encode_chunk(frames)
        chunks = torch.split(frames, chunk_sz, dim=0)
        if self.training and torch.is_grad_enabled():
            import torch.utils.checkpoint as cp
            cls_list = [cp.checkpoint(self._encode_chunk, c, use_reentrant=False) for c in chunks]
        else:
            cls_list = [self._encode_chunk(c) for c in chunks]
        return torch.cat(cls_list, dim=0)

    def forward(self, x: torch.Tensor, **kwargs):
        if x.dim() == 5:
            B, T, C, H, W = x.shape
            flat_frames = x.reshape(B * T, C, H, W)
            frame_cls = self._encode_frames(flat_frames).reshape(B, T, self.in_planes)
            global_feat = self.temporal_pool(frame_cls)
        elif x.dim() == 4:
            frame_cls = self._encode_frames(x).unsqueeze(1)  # (B, 1, D)
            global_feat = self.temporal_pool(frame_cls)
        else:
            raise ValueError(f"Unexpected input shape for SCLVideoModel: {tuple(x.shape)}")

        if self.scnorm_before_bn is not None:
            global_feat = self.scnorm_before_bn(global_feat)

        feat = self.bottleneck(global_feat)

        if self.training:
            cls_score = self.classifier(feat)
            return cls_score, global_feat

        if self.neck_feat == "after":
            return feat
        return global_feat
