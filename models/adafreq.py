from functools import partial
import math
from pathlib import Path
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .reid_model import TemporalPool

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ADAFREQ_ROOT = PROJECT_ROOT / "AdaFreq"
SCL_ROOT = PROJECT_ROOT / "ECCV-26-SCL"


def _load_adafreq_components():
    """Import AdaFreq config and weight-init utilities without namespace collisions."""
    if str(ADAFREQ_ROOT) not in sys.path:
        sys.path.insert(0, str(ADAFREQ_ROOT))

    saved_modules = {}
    for mod_name in ("config", "config.defaults"):
        if mod_name in sys.modules:
            saved_modules[mod_name] = sys.modules.pop(mod_name)

    try:
        from config.defaults import _C as adafreq_default_cfg
        return adafreq_default_cfg
    finally:
        for k, v in saved_modules.items():
            sys.modules[k] = v


_ADAFREQ_CFG_PROTO = _load_adafreq_components()


def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find("Linear") != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode="fan_out")
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find("Conv") != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode="fan_in")
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find("BatchNorm") != -1:
        if m.affine:
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)


def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find("Linear") != -1:
        nn.init.normal_(m.weight, std=0.001)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)


def _no_grad_trunc_normal_(tensor, mean, std, a, b):
    def norm_cdf(x):
        return (1.0 + math.erf(x / math.sqrt(2.0))) / 2.0

    with torch.no_grad():
        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)
        tensor.uniform_(2 * l - 1, 2 * u - 1)
        tensor.erfinv_()
        tensor.mul_(std * math.sqrt(2.0))
        tensor.add_(mean)
        tensor.clamp_(min=a, max=b)
        return tensor


def trunc_normal_(tensor, mean=0.0, std=1.0, a=-2.0, b=2.0):
    return _no_grad_trunc_normal_(tensor, mean, std, a, b)


def drop_path(x, drop_prob: float = 0.0, training: bool = False):
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()
    return x.div(keep_prob) * random_tensor


class DropPath(nn.Module):
    def __init__(self, drop_prob=None):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.0):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class AdaFreqAttention(nn.Module):
    """
    Exact Attention module from AdaFreq/model/backbones/vit_pytorch.py,
    capturing the self-attention map natively on GPU without the broken
    Python 3.13 `bytecode` package.
    """

    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.head_dim = head_dim
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.capture_attn = False
        self.last_cls_attn = None

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        if self.capture_attn:
            attn = (q @ k.transpose(-2, -1)) * self.scale
            attn = attn.softmax(dim=-1)
            # Capture CLS-to-patch attention (B, num_heads, N-1) detached on GPU
            self.last_cls_attn = attn[:, :, 0, 1:].detach()
            attn = self.attn_drop(attn)
            x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        else:
            # Memory-efficient Scaled Dot-Product Attention for blocks 0..10 and Pass 2
            x = F.scaled_dot_product_attention(
                q, k, v,
                dropout_p=self.attn_drop.p if self.training else 0.0
            ).transpose(1, 2).reshape(B, N, C)

        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class AdaFreqBlock(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.0,
        qkv_bias=False,
        qk_scale=None,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = AdaFreqAttention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class PatchEmbedOverlap(nn.Module):
    """Image to Patch Embedding with overlapping patches (exact port from AdaFreq)."""

    def __init__(self, img_size=(256, 256), patch_size=(16, 16), stride_size=(12, 12), in_chans=3, embed_dim=768):
        super().__init__()
        img_size = (int(img_size[0]), int(img_size[1]))
        patch_size = (int(patch_size[0]), int(patch_size[1])) if isinstance(patch_size, (tuple, list)) else (int(patch_size), int(patch_size))
        stride_size = (int(stride_size[0]), int(stride_size[1])) if isinstance(stride_size, (tuple, list)) else (int(stride_size), int(stride_size))

        self.num_x = (img_size[1] - patch_size[1]) // stride_size[1] + 1
        self.num_y = (img_size[0] - patch_size[0]) // stride_size[0] + 1
        print(f"[AdaFreq] using stride: {stride_size}, and patch number is num_y={self.num_y} * num_x={self.num_x}")
        self.num_patches = self.num_x * self.num_y
        self.img_size = img_size
        self.patch_size = patch_size

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=stride_size)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))

    def forward(self, x):
        B, C, H, W = x.shape
        assert H == self.img_size[0] and W == self.img_size[1], (
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        )
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)
        return x


def resize_pos_embed(posemb, posemb_new, hight, width):
    posemb_token, posemb_grid = posemb[:, :1], posemb[0, 1:]
    gs_old = int(math.sqrt(len(posemb_grid)))
    print(f"[AdaFreq] Resized position embedding from size:{posemb.shape} to size:{posemb_new.shape} with height:{hight} width:{width}")
    posemb_grid = posemb_grid.reshape(1, gs_old, gs_old, -1).permute(0, 3, 1, 2)
    posemb_grid = F.interpolate(posemb_grid, size=(hight, width), mode="bilinear")
    posemb_grid = posemb_grid.permute(0, 2, 3, 1).reshape(1, hight * width, -1)
    posemb = torch.cat([posemb_token, posemb_grid], dim=1)
    return posemb


class AdaFreqTransReID(nn.Module):
    """
    Exact TransReID ViT backbone from AdaFreq/model/backbones/vit_pytorch.py,
    supporting adaptive top-mu token selection (`atten`) in Pass 2.
    """

    def __init__(
        self,
        img_size=(256, 256),
        patch_size=16,
        stride_size=(12, 12),
        in_chans=3,
        num_classes=1000,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.1,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        local_feature=True,
        mu=0.5,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim
        self.local_feature = local_feature

        self.patch_embed = PatchEmbedOverlap(
            img_size=img_size,
            patch_size=patch_size,
            stride_size=stride_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
        )
        num_patches = self.patch_embed.num_patches
        self.num_keep = int(round(mu * num_patches)) if num_patches != 441 else 220

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_embed_1 = nn.Parameter(torch.zeros(1, self.num_keep + 1, embed_dim))

        self.pos_drop = nn.Dropout(p=drop_rate)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]

        self.blocks = nn.ModuleList([
            AdaFreqBlock(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[i],
                norm_layer=norm_layer,
            )
            for i in range(depth)
        ])
        # Only the last block (blocks[11]) needs to capture the attention map
        self.blocks[-1].attn.capture_attn = True

        self.norm = norm_layer(embed_dim)
        self.fc = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()

        trunc_normal_(self.cls_token, std=0.02)
        trunc_normal_(self.pos_embed, std=0.02)
        trunc_normal_(self.pos_embed_1, std=0.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def _forward_blocks(self, x):
        import torch.utils.checkpoint as cp
        if self.training and torch.is_grad_enabled():
            for blk in self.blocks[:-1]:
                x = cp.checkpoint(blk, x, use_reentrant=False)
            x = self.blocks[-1](x)
        else:
            for blk in self.blocks:
                x = blk(x)
        return x

    def forward_features(self, x, atten=None):
        B = x.shape[0]
        x = self.patch_embed(x)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        if atten is not None:
            # Vectorized gather of selected patch indices per frame: atten has shape (B, num_keep + 1)
            idx_expanded = atten.unsqueeze(-1).expand(-1, -1, x.size(-1))
            x = torch.gather(x, dim=1, index=idx_expanded)
            x = x + self.pos_embed_1
        else:
            x = x + self.pos_embed

        x = self.pos_drop(x)
        x = self._forward_blocks(x)
        x = self.norm(x)

        if self.local_feature:
            return x
        return x[:, 0]

    def forward(self, x, atten=None):
        return self.forward_features(x, atten=atten)

    def load_param(self, model_path):
        param_dict = torch.load(model_path, map_location="cpu")
        if "model" in param_dict:
            param_dict = param_dict["model"]
        if "state_dict" in param_dict:
            param_dict = param_dict["state_dict"]
        for k, v in param_dict.items():
            if "head" in k or "dist" in k:
                continue
            if "patch_embed.proj.weight" in k and len(v.shape) < 4:
                O, I, H, W = self.patch_embed.proj.weight.shape
                v = v.reshape(O, -1, H, W)
            elif k == "pos_embed" and v.shape != self.pos_embed.shape:
                if "distilled" in str(model_path):
                    v = torch.cat([v[:, 0:1], v[:, 2:]], dim=1)
                v = resize_pos_embed(v, self.pos_embed, self.patch_embed.num_y, self.patch_embed.num_x)
            if k in self.state_dict() and self.state_dict()[k].shape == v.shape:
                self.state_dict()[k].copy_(v)


class TorchHighFrequencyFilter(nn.Module):
    """
    Batched GPU implementation of AdaFreq/processor/high_fre.py (HighFrequencyFilter).
    Performs the exact mathematical operations:
      1. Convert RGB to grayscale
      2. 2D FFT + fftshift -> Gaussian high-pass filter (D=70) -> IFFT + min-max [0, 255]
      3. Frequency-domain CutMix (`bbox` with lam ~ Beta(1, 1) * 0.5) between original spectrum and high-pass spectrum
      4. IFFT -> min-max normalization -> replicate to 3 channels and normalize to match input distribution.
    """

    def __init__(self, D: float = 70.0, pixel_mean=(0.5, 0.5, 0.5), pixel_std=(0.5, 0.5, 0.5)):
        super().__init__()
        self.D = float(D)
        self.register_buffer("pixel_mean", torch.tensor(pixel_mean, dtype=torch.float32).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("pixel_std", torch.tensor(pixel_std, dtype=torch.float32).view(1, 3, 1, 1), persistent=False)
        self._cached_grid = {}

    def _get_high_pass_template(self, H: int, W: int, device: torch.device) -> torch.Tensor:
        key = (H, W, device)
        if key not in self._cached_grid:
            x = torch.arange(H, dtype=torch.float32, device=device).unsqueeze(1)
            y = torch.arange(W, dtype=torch.float32, device=device).unsqueeze(0)
            cx = int((H - 1) / 2)
            cy = int((W - 1) / 2)
            dis_square = (x - cx) ** 2 + (y - cy) ** 2
            template = 1.0 - torch.exp(-dis_square / (2.0 * (self.D ** 2)))
            self._cached_grid[key] = template.unsqueeze(0)  # (1, H, W)
        return self._cached_grid[key]

    @torch.no_grad()
    def apply_filter(self, images: torch.Tensor) -> torch.Tensor:
        """
        Args:
            images: Normalized RGB tensor of shape (N, 3, H, W) on GPU.
        Returns:
            high_freq_images: Normalized 3-channel high-frequency tensor of shape (N, 3, H, W) on GPU.
        """
        orig_dtype = images.dtype
        device = images.device
        N, C, H, W = images.shape

        # Unnormalize from [-1, 1] back to [0, 255] RGB so grayscale & FFT match uint8 image scale
        mean = self.pixel_mean.to(device=device, dtype=torch.float32)
        std = self.pixel_std.to(device=device, dtype=torch.float32)
        rgb_01 = (images.float() * std + mean).clamp(0.0, 1.0)
        gray = (0.299 * rgb_01[:, 0] + 0.587 * rgb_01[:, 1] + 0.114 * rgb_01[:, 2]) * 255.0  # (N, H, W)

        # 1. 2D FFT & Gaussian high-pass filter (D=70)
        f = torch.fft.fft2(gray)
        fshift1 = torch.fft.fftshift(f, dim=(-2, -1))
        template = self._get_high_pass_template(H, W, device)
        high_parts_fshift = fshift1 * template

        high_parts_img = torch.fft.ifft2(torch.fft.ifftshift(high_parts_fshift, dim=(-2, -1))).abs()
        amin = high_parts_img.amin(dim=(-2, -1), keepdim=True)
        amax = high_parts_img.amax(dim=(-2, -1), keepdim=True)
        use_img = ((high_parts_img - amin + 1e-5) / (amax - amin + 1e-5)) * 255.0

        # 2. Second FFT + Frequency CutMix with lam ~ Beta(1.0, 1.0) * 0.5
        fshift2 = torch.fft.fftshift(torch.fft.fft2(use_img), dim=(-2, -1)).clone()
        lam = np.random.beta(1.0, 1.0) * 0.5
        cut_rat = math.sqrt(1.0 - lam)
        cut_h = int(H * cut_rat)
        cut_w = int(W * cut_rat)
        cx = np.random.randint(H)
        cy = np.random.randint(W)
        bbx1 = max(0, min(H, cx - cut_h // 2))
        bby1 = max(0, min(W, cy - cut_w // 2))
        bbx2 = max(0, min(H, cx + cut_h // 2))
        bby2 = max(0, min(W, cy + cut_w // 2))
        fshift2[:, bbx1:bbx2, bby1:bby2] = fshift1[:, bbx1:bbx2, bby1:bby2]

        # 3. IFFT + MinMax normalize to [0, 1] + Re-normalize with pixel_mean / pixel_std
        mixed_img = torch.fft.ifft2(torch.fft.ifftshift(fshift2, dim=(-2, -1))).abs()
        m_min = mixed_img.amin(dim=(-2, -1), keepdim=True)
        m_max = mixed_img.amax(dim=(-2, -1), keepdim=True)
        norm_01 = (mixed_img - m_min) / (m_max - m_min + 1e-6)  # (N, H, W) in [0, 1]

        out_3c = norm_01.unsqueeze(1).expand(-1, 3, -1, -1)
        out_norm = (out_3c - mean) / std
        return out_norm.to(orig_dtype)


def build_adafreq_cfg(cfg):
    """Build the AdaFreq YACS config matching the exact paper parameters."""
    af_cfg = _ADAFREQ_CFG_PROTO.clone()
    af_cfg.defrost()
    af_cfg.MODEL.DIST_TRAIN = False
    af_cfg.MODEL.NAME = "transformer"
    af_cfg.MODEL.TRANSFORMER_TYPE = "vit_base_patch16_224_TransReID"
    af_cfg.MODEL.STRIDE_SIZE = list(getattr(cfg, "adafreq_stride_size", [12, 12]))
    af_cfg.MODEL.JPM = True
    af_cfg.MODEL.RE_ARRANGE = False
    af_cfg.MODEL.NO_MARGIN = True
    af_cfg.MODEL.IF_LABELSMOOTH = "off"
    af_cfg.MODEL.IF_WITH_CENTER = "no"
    af_cfg.MODEL.METRIC_LOSS_TYPE = "triplet"
    af_cfg.MODEL.ID_LOSS_WEIGHT = 1.0
    af_cfg.MODEL.TRIPLET_LOSS_WEIGHT = 1.0

    pretrain_path = SCL_ROOT / "pretrained" / "jx_vit_base_p16_224-80ecf9dd.pth"
    af_cfg.MODEL.PRETRAIN_PATH = str(pretrain_path)

    img_size = tuple(getattr(cfg, "img_size", (256, 256)))
    af_cfg.INPUT.SIZE_TRAIN = list(img_size)
    af_cfg.INPUT.SIZE_TEST = list(img_size)
    af_cfg.INPUT.PIXEL_MEAN = [0.5, 0.5, 0.5]
    af_cfg.INPUT.PIXEL_STD = [0.5, 0.5, 0.5]
    af_cfg.INPUT.PADDING = int(getattr(cfg, "aug_pad", 10))
    af_cfg.INPUT.RE_PROB = float(getattr(cfg, "re_prob", 0.0))
    af_cfg.INPUT.RO_DEGREE = 15

    af_cfg.DATALOADER.SAMPLER = "softmax_triplet"
    af_cfg.DATALOADER.NUM_INSTANCE = int(getattr(cfg, "k", 4))

    af_cfg.SOLVER.OPTIMIZER_NAME = "SGD"
    af_cfg.SOLVER.MAX_EPOCHS = int(getattr(cfg, "epochs", 150))
    af_cfg.SOLVER.BASE_LR = float(getattr(cfg, "lr", 0.001))
    af_cfg.SOLVER.IMS_PER_BATCH = int(getattr(cfg, "batch_size", 32))
    af_cfg.SOLVER.WARMUP_METHOD = "linear"
    af_cfg.SOLVER.WARMUP_EPOCHS = int(getattr(cfg, "warmup_epochs", 5))
    af_cfg.SOLVER.LARGE_FC_LR = False
    af_cfg.SOLVER.WEIGHT_DECAY = float(getattr(cfg, "weight_decay", 1e-4))
    af_cfg.SOLVER.WEIGHT_DECAY_BIAS = float(getattr(cfg, "weight_decay", 1e-4))
    af_cfg.SOLVER.BIAS_LR_FACTOR = 2
    af_cfg.SOLVER.MOMENTUM = 0.9

    af_cfg.TEST.NECK_FEAT = "after"
    af_cfg.TEST.FEAT_NORM = "yes"
    af_cfg.freeze()
    return af_cfg


class AdaFreqVideoModel(nn.Module):
    """
    Video adaptation of AdaFreq ([ECCV 2024] Adaptive High-Frequency Transformer
    for Diverse Wildlife Re-Identification).

    Training:
      - Pass 1 (Original RGB video frames x):
          Extracts all patch tokens from `self.base`, pools frame CLS tokens via
          `self.temporal_pool` ('mean' or 'attention') -> `global_feat` -> `bottleneck` -> `classifier`.
          Finds top-mu (mu=0.5 -> 220) attended patch indices (`need`) per frame from the last
          ViT layer's CLS attention map, and extracts `ori_part` patch tokens.
      - Pass 2 (High-frequency filtered video frames y):
          Feeds `y` with `atten=need` into `self.base` (processing only CLS + top-220 patches),
          pools high-frequency frame CLS tokens via `self.temporal_pool` -> `high_feat` -> `bottleneck_1` -> `classifier_1`,
          and returns `[cls_score, cls_score_1], [global_feat, high_feat, ori_part, high_feat_1]`.
    Inference (`model.eval()`):
      - Single-pass using only the original RGB frames (`bottleneck(temporal_pool(CLS))`), matching the paper.
    """

    def __init__(self, cfg):
        super().__init__()
        self.is_adafreq = True
        self.af_cfg = build_adafreq_cfg(cfg)
        self.pooling_type = str(getattr(cfg, "pooling_type", "mean") or "mean").lower()
        if self.pooling_type == "attn":
            self.pooling_type = "attention"
        self.chunk_size = int(getattr(cfg, "chunk_size", 32) or 32)
        self.num_classes = int(getattr(cfg, "num_classes", 0) or 0)
        self.neck_feat = self.af_cfg.TEST.NECK_FEAT
        self.mu = float(getattr(cfg, "adafreq_mu", 0.5))

        img_size = tuple(self.af_cfg.INPUT.SIZE_TRAIN)
        stride_size = tuple(self.af_cfg.MODEL.STRIDE_SIZE)

        # 1. TransReID ViT-B/16 with overlapping stride [12, 12]
        self.base = AdaFreqTransReID(
            img_size=img_size,
            patch_size=16,
            stride_size=stride_size,
            embed_dim=768,
            depth=12,
            num_heads=12,
            mlp_ratio=4.0,
            qkv_bias=True,
            drop_path_rate=self.af_cfg.MODEL.DROP_PATH,
            drop_rate=self.af_cfg.MODEL.DROP_OUT,
            attn_drop_rate=self.af_cfg.MODEL.ATT_DROP_RATE,
            local_feature=True,
            mu=self.mu,
        )
        self.in_planes = 768
        self.num_keep = self.base.num_keep

        pretrain_path = Path(self.af_cfg.MODEL.PRETRAIN_PATH)
        if pretrain_path.exists():
            self.base.load_param(str(pretrain_path))
            print(f"[AdaFreq] Loaded ImageNet pretrained ViT weights from {pretrain_path}")
        else:
            print(f"[AdaFreq] WARNING: Pretrained ViT file not found at {pretrain_path}")

        # 2. Temporal Pooling across frames ('mean' or 'attention')
        self.temporal_pool = TemporalPool(self.in_planes, mode=self.pooling_type)

        # 3. Dual Classifiers & BNNecks (Original branch + High-Frequency branch)
        self.classifier = nn.Linear(self.in_planes, max(1, self.num_classes), bias=False)
        self.classifier.apply(weights_init_classifier)
        self.classifier_1 = nn.Linear(self.in_planes, max(1, self.num_classes), bias=False)
        self.classifier_1.apply(weights_init_classifier)

        self.bottleneck = nn.BatchNorm1d(self.in_planes)
        self.bottleneck.bias.requires_grad_(False)
        self.bottleneck.apply(weights_init_kaiming)

        self.bottleneck_1 = nn.BatchNorm1d(self.in_planes)
        self.bottleneck_1.bias.requires_grad_(False)
        self.bottleneck_1.apply(weights_init_kaiming)

    def _select_top_patches(self, cls_attn: torch.Tensor) -> torch.Tensor:
        """
        Given CLS-to-patch attention of shape (N, num_heads, N_patches),
        compute head-averaged attention and return indices `need` of shape (N, 1 + num_keep)
        where `need[:, 0] == 0` (CLS token) and `need[:, 1:]` are the top-`num_keep` patch token indices in [1..N_patches].
        """
        mean_attn = cls_attn.mean(dim=1)  # (N, N_patches)
        _, topk_idx = torch.topk(mean_attn, k=self.num_keep, dim=1, largest=True, sorted=False)
        patch_positions = topk_idx + 1  # shift by +1 because index 0 is CLS token
        cls_zero = torch.zeros((mean_attn.size(0), 1), dtype=patch_positions.dtype, device=patch_positions.device)
        need = torch.cat([cls_zero, patch_positions], dim=1)  # (N, 1 + num_keep)
        return need

    def _forward_pass1_chunks(self, flat_frames: torch.Tensor):
        """
        Run Pass 1 on `flat_frames` (N, C, H, W) in chunks to bound peak memory.
        Returns:
            cls_tokens: (N, 768)
            ori_part: (N, 1 + num_keep, 768) or None if not training
            need: (N, 1 + num_keep) or None if not training
        """
        chunk_sz = min(self.chunk_size, 8) if self.training else self.chunk_size
        chunks = torch.split(flat_frames, chunk_sz, dim=0)

        cls_list = []
        ori_part_list = []
        need_list = []

        for c in chunks:
            tokens = self.base(c, atten=None)  # (n, 442, 768)
            cls_list.append(tokens[:, 0])
            if self.training:
                cls_attn = self.base.blocks[-1].attn.last_cls_attn  # (n, 12, 441)
                need_c = self._select_top_patches(cls_attn)  # (n, 221)
                idx_exp = need_c.unsqueeze(-1).expand(-1, -1, tokens.size(-1))
                ori_c = torch.gather(tokens, dim=1, index=idx_exp)  # (n, 221, 768)
                ori_part_list.append(ori_c)
                need_list.append(need_c)
            self.base.blocks[-1].attn.last_cls_attn = None

        cls_tokens = torch.cat(cls_list, dim=0)
        if self.training:
            return cls_tokens, torch.cat(ori_part_list, dim=0), torch.cat(need_list, dim=0)
        return cls_tokens, None, None

    def _forward_pass2_chunks(self, flat_high_frames: torch.Tensor, need: torch.Tensor):
        """
        Run Pass 2 on high-frequency frames `flat_high_frames` (N, C, H, W) with selected patch indices `need` (N, 221).
        Returns:
            high_cls_tokens: (N, 768)
            high_feat_1: (N, 1 + num_keep, 768)
        """
        chunk_sz = min(self.chunk_size, 8)
        f_chunks = torch.split(flat_high_frames, chunk_sz, dim=0)
        n_chunks = torch.split(need, chunk_sz, dim=0)

        high_cls_list = []
        high_part_list = []

        for fc, nc in zip(f_chunks, n_chunks):
            high_tokens = self.base(fc, atten=nc)  # (n, 221, 768)
            high_cls_list.append(high_tokens[:, 0])
            high_part_list.append(high_tokens)
            self.base.blocks[-1].attn.last_cls_attn = None

        return torch.cat(high_cls_list, dim=0), torch.cat(high_part_list, dim=0)

    def forward(self, x: torch.Tensor, y: torch.Tensor = None, **kwargs):
        if x.dim() == 5:
            B, T, C, H, W = x.shape
            flat_x = x.reshape(B * T, C, H, W)
        elif x.dim() == 4:
            B, C, H, W = x.shape
            T = 1
            flat_x = x
        else:
            raise ValueError(f"Unexpected input shape for AdaFreqVideoModel: {tuple(x.shape)}")

        if self.training and y is not None:
            flat_y = y.reshape(B * T, C, H, W) if y.dim() == 5 else y
            frame_cls, ori_part, need = self._forward_pass1_chunks(flat_x)
            global_feat = self.temporal_pool(frame_cls.reshape(B, T, self.in_planes))
            feat = self.bottleneck(global_feat)
            cls_score = self.classifier(feat)

            high_frame_cls, high_feat_1 = self._forward_pass2_chunks(flat_y, need)
            high_feat = self.temporal_pool(high_frame_cls.reshape(B, T, self.in_planes))
            feat_1 = self.bottleneck_1(high_feat)
            cls_score_1 = self.classifier_1(feat_1)

            return [cls_score, cls_score_1], [global_feat, high_feat, ori_part, high_feat_1]

        frame_cls, _, _ = self._forward_pass1_chunks(flat_x)
        global_feat = self.temporal_pool(frame_cls.reshape(B, T, self.in_planes))
        feat = self.bottleneck(global_feat)

        if self.training:
            cls_score = self.classifier(feat)
            return cls_score, global_feat

        if self.neck_feat == "after":
            return feat
        return global_feat
