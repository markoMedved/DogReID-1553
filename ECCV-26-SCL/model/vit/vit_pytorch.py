import math
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.modules.freq_norm_vit import FreqNormOnTokens

from .common import PatchEmbed_overlap, ResidualAttentionBlock


class TransReID(nn.Module):
    def __init__(
        self,
        img_size=224,
        patch_size=16,
        stride_size=16,
        in_chans=3,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.0,
        qkv_bias=False,
        qk_scale=None,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
        norm_layer=nn.LayerNorm,
        freq_norm_enabled=False,
        freq_norm_kind="scfg",
        freq_norm_base="in",
        freq_norm_positions=None,
        freq_norm_pos0_kind="scfg",
        scnorm_fg_ratio=0.4,
        scnorm_fg_mask_temp=0.15,
        scfg_constrained=True,
        scfg_spatial_split=False,
        scfg_min_fg_norm=0.05,
        scfg_min_bg_gap=0.10,
        scfg_max_norm=0.95,
        scfg_warmup_epochs=0,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.patch_embed = PatchEmbed_overlap(
            img_size=img_size,
            patch_size=patch_size,
            stride_size=stride_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
        )
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.blocks = nn.Sequential(
            *[
                ResidualAttentionBlock(
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
            ]
        )
        self.norm = norm_layer(embed_dim)

        self.freq_norm_enabled = bool(freq_norm_enabled)
        self.freq_norm_positions = set(freq_norm_positions or [])
        self.freq_norms = nn.ModuleDict()
        if self.freq_norm_enabled:
            for pos in self.freq_norm_positions:
                kind = freq_norm_pos0_kind if pos == 0 else freq_norm_kind
                self.freq_norms[f"pos_{pos}"] = FreqNormOnTokens(
                    C=embed_dim,
                    H=getattr(self.patch_embed, "num_y", 14),
                    W=getattr(self.patch_embed, "num_x", 14),
                    kind=kind,
                    base=freq_norm_base,
                    with_cls=True,
                    fg_ratio=scnorm_fg_ratio,
                    fg_mask_temp=scnorm_fg_mask_temp,
                    scfg_constrained=scfg_constrained,
                    scfg_spatial_split=scfg_spatial_split,
                    scfg_min_fg_norm=scfg_min_fg_norm,
                    scfg_min_bg_gap=scfg_min_bg_gap,
                    scfg_max_norm=scfg_max_norm,
                    scfg_warmup_epochs=scfg_warmup_epochs,
                )
                print(f"FreqNorm enabled at pos_{pos}: kind={kind}, base={freq_norm_base}")

        trunc_normal_(self.cls_token, std=0.02)
        trunc_normal_(self.pos_embed, std=0.02)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.LayerNorm):
            nn.init.constant_(module.bias, 0)
            nn.init.constant_(module.weight, 1.0)

    def forward_main(self, x):
        batch_size = x.shape[0]
        x = self.patch_embed(x)
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        if self.freq_norm_enabled and 0 in self.freq_norm_positions:
            x = self.freq_norms["pos_0"](x)

        block_positions = self.freq_norm_positions - {0}
        if self.freq_norm_enabled and block_positions:
            for i, block in enumerate(self.blocks, start=1):
                x = block(x)
                if i in self.freq_norm_positions:
                    x = self.freq_norms[f"pos_{i}"](x)
        else:
            x = self.blocks(x)

        return self.norm(x)

    def forward(self, forward_type="main", *args, **kwargs):
        if forward_type != "main":
            raise ValueError("TransReID only supports forward_type='main' in this release.")
        return self.forward_main(*args, **kwargs)

    def load_param(self, model_path):
        param_dict = torch.load(model_path, map_location="cpu")
        if "model" in param_dict:
            param_dict = param_dict["model"]
        if "state_dict" in param_dict:
            param_dict = param_dict["state_dict"]

        own_state = self.state_dict()
        for key, value in param_dict.items():
            if "head" in key or "dist" in key:
                continue
            clean_key = key.replace("module.", "")
            if "patch_embed.proj.weight" in clean_key and len(value.shape) < 4:
                out_channels, in_channels, height, width = self.patch_embed.proj.weight.shape
                value = value.reshape(out_channels, -1, height, width)
            elif clean_key == "pos_embed" and value.shape != self.pos_embed.shape:
                value = resize_pos_embed(
                    value,
                    self.pos_embed,
                    self.patch_embed.num_y,
                    self.patch_embed.num_x,
                )
            if clean_key in own_state and own_state[clean_key].shape == value.shape:
                own_state[clean_key].copy_(value)


def vit_base_patch16_224_TransReID(cfg, **kwargs):
    return TransReID(
        img_size=cfg.INPUT.SIZE_TRAIN,
        patch_size=16,
        stride_size=cfg.MODEL.STRIDE_SIZE,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4,
        qkv_bias=True,
        drop_path_rate=cfg.MODEL.DROP_PATH,
        drop_rate=cfg.MODEL.DROP_OUT,
        attn_drop_rate=cfg.MODEL.ATT_DROP_RATE,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        freq_norm_enabled=cfg.CHANGE.METHODS.FREQ_NORM_ENABLED,
        freq_norm_kind=cfg.CHANGE.METHODS.FREQ_NORM_KIND,
        freq_norm_base=cfg.CHANGE.METHODS.FREQ_NORM_BASE,
        freq_norm_positions=cfg.CHANGE.METHODS.FREQ_NORM_POSITIONS,
        freq_norm_pos0_kind=cfg.CHANGE.METHODS.FREQ_NORM_POS0_KIND,
        scnorm_fg_ratio=cfg.CHANGE.METHODS.SCNORM_FG_RATIO,
        scnorm_fg_mask_temp=cfg.CHANGE.METHODS.SCNORM_FG_MASK_TEMP,
        scfg_constrained=cfg.CHANGE.METHODS.SCFG_CONSTRAINED,
        scfg_spatial_split=cfg.CHANGE.METHODS.SCFG_SPATIAL_SPLIT,
        scfg_min_fg_norm=cfg.CHANGE.METHODS.SCFG_MIN_FG_NORM,
        scfg_min_bg_gap=cfg.CHANGE.METHODS.SCFG_MIN_BG_GAP,
        scfg_max_norm=cfg.CHANGE.METHODS.SCFG_MAX_NORM,
        scfg_warmup_epochs=cfg.CHANGE.METHODS.SCFG_WARMUP_EPOCHS,
        **kwargs,
    )


def resize_pos_embed(posemb, posemb_new, height, width):
    ntok_new = posemb_new.shape[1] - 1
    posemb_token, posemb_grid = posemb[:, :1], posemb[0, 1:]
    gs_old = int(math.sqrt(len(posemb_grid)))
    posemb_grid = posemb_grid.reshape(1, gs_old, gs_old, -1).permute(0, 3, 1, 2)
    posemb_grid = F.interpolate(posemb_grid, size=(height, width), mode="bilinear")
    posemb_grid = posemb_grid.permute(0, 2, 3, 1).reshape(1, ntok_new, -1)
    return torch.cat([posemb_token, posemb_grid], dim=1)


def _no_grad_trunc_normal_(tensor, mean, std, a, b):
    def norm_cdf(x):
        return (1.0 + math.erf(x / math.sqrt(2.0))) / 2.0

    with torch.no_grad():
        lower = norm_cdf((a - mean) / std)
        upper = norm_cdf((b - mean) / std)
        tensor.uniform_(2 * lower - 1, 2 * upper - 1)
        tensor.erfinv_()
        tensor.mul_(std * math.sqrt(2.0))
        tensor.add_(mean)
        tensor.clamp_(min=a, max=b)
        return tensor


def trunc_normal_(tensor, mean=0.0, std=1.0, a=-2.0, b=2.0):
    return _no_grad_trunc_normal_(tensor, mean, std, a, b)
