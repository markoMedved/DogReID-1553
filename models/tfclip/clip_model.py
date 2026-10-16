import math
import os
import urllib.request
import hashlib
import warnings
from collections import OrderedDict
import torch
import torch.nn as nn
import torch.nn.functional as F

CLIP_VITB16_URL = "https://openaipublic.azureedge.net/clip/models/5806e77cd80f8b59890b7e101eabd078d9fb84e6937f9e85e4ecb61988df416f/ViT-B-16.pt"
CLIP_VITB16_SHA256 = "5806e77cd80f8b59890b7e101eabd078d9fb84e6937f9e85e4ecb61988df416f"


class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16/bf16 in float32."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)


class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


class ResidualAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask

    def attention(self, x: torch.Tensor):
        mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        return self.attn(x, x, x, need_weights=False, attn_mask=mask)[0]

    def forward(self, x: torch.Tensor):
        x = x + self.attention(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class Transformer(nn.Module):
    def __init__(self, width: int, layers: int, heads: int, attn_mask: torch.Tensor = None):
        super().__init__()
        self.width = width
        self.layers = layers
        self.resblocks = nn.Sequential(*[ResidualAttentionBlock(width, heads, attn_mask) for _ in range(layers)])

    def forward(self, x: torch.Tensor):
        return self.resblocks(x)


class VisionTransformer(nn.Module):
    def __init__(
        self,
        h_resolution: int = 14,
        w_resolution: int = 14,
        patch_size: int = 16,
        stride_size: int = 16,
        width: int = 768,
        layers: int = 12,
        heads: int = 12,
        output_dim: int = 512,
    ):
        super().__init__()
        self.h_resolution = h_resolution
        self.w_resolution = w_resolution
        self.output_dim = output_dim
        self.conv1 = nn.Conv2d(
            in_channels=3,
            out_channels=width,
            kernel_size=patch_size,
            stride=stride_size,
            bias=False
        )

        scale = width ** -0.5
        self.class_embedding = nn.Parameter(scale * torch.randn(width))
        self.positional_embedding = nn.Parameter(scale * torch.randn(h_resolution * w_resolution + 1, width))
        self.ln_pre = LayerNorm(width)
        self.transformer = Transformer(width, layers, heads)
        self.ln_post = LayerNorm(width)
        self.proj = nn.Parameter(scale * torch.randn(width, output_dim))

    def forward(self, x: torch.Tensor, cv_emb: torch.Tensor = None):
        # x: [B*T, 3, H, W]
        x = self.conv1(x)  # [*, width, grid_h, grid_w]
        x = x.reshape(x.shape[0], x.shape[1], -1)  # [*, width, grid**2]
        x = x.permute(0, 2, 1)  # [*, grid**2, width]

        cls_token = self.class_embedding.to(x.dtype) + torch.zeros(
            x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device
        )
        x = torch.cat([cls_token, x], dim=1)  # [*, grid**2 + 1, width]

        if cv_emb is not None:
            x[:, 0] = x[:, 0] + cv_emb

        x = x + self.positional_embedding.to(x.dtype)
        x = self.ln_pre(x)

        x = x.permute(1, 0, 2)  # [L, N, D]
        x11 = self.transformer.resblocks[:11](x)
        x12 = self.transformer.resblocks[11](x11)
        x12 = x12.permute(1, 0, 2)  # [N, L, D]
        x12 = self.ln_post(x12)

        xproj = x12 @ self.proj if self.proj is not None else None
        return x12, xproj


def resize_pos_embed(posemb: torch.Tensor, posemb_new: torch.Tensor, height: int, width: int):
    """Rescale position embeddings grid when resolution changes."""
    posemb_token, posemb_grid = posemb[:1], posemb[1:]
    gs_old = int(math.sqrt(len(posemb_grid)))
    if gs_old == height and gs_old == width:
        return posemb

    posemb_grid = posemb_grid.reshape(1, gs_old, gs_old, -1).permute(0, 3, 1, 2)
    posemb_grid = F.interpolate(posemb_grid, size=(height, width), mode='bilinear', align_corners=False)
    posemb_grid = posemb_grid.permute(0, 2, 3, 1).reshape(1, height * width, -1)
    posemb = torch.cat([posemb_token, posemb_grid.squeeze(0)], dim=0)
    return posemb


def download_clip_weights(cache_dir: str = os.path.expanduser("~/.cache/clip")) -> str:
    os.makedirs(cache_dir, exist_ok=True)
    target_path = os.path.join(cache_dir, "ViT-B-16.pt")

    if os.path.isfile(target_path):
        return target_path

    print(f"Downloading CLIP ViT-B/16 to {target_path} ...")
    with urllib.request.urlopen(CLIP_VITB16_URL) as response, open(target_path, "wb") as out_file:
        while True:
            chunk = response.read(16384)
            if not chunk:
                break
            out_file.write(chunk)
    return target_path


def load_clip_vitb16(h_resolution: int = 14, w_resolution: int = 14, patch_size: int = 16, stride_size: int = 16) -> VisionTransformer:
    model_path = download_clip_weights()
    try:
        raw_model = torch.jit.load(model_path, map_location="cpu").eval()
        full_sd = raw_model.state_dict()
    except Exception:
        full_sd = torch.load(model_path, map_location="cpu")

    vis_sd = {}
    for k, v in full_sd.items():
        if k.startswith("visual."):
            vis_sd[k[len("visual."):]] = v.float()

    model = VisionTransformer(
        h_resolution=h_resolution,
        w_resolution=w_resolution,
        patch_size=patch_size,
        stride_size=stride_size,
        width=768,
        layers=12,
        heads=12,
        output_dim=512
    )

    if "positional_embedding" in vis_sd:
        vis_sd["positional_embedding"] = resize_pos_embed(
            vis_sd["positional_embedding"],
            model.positional_embedding,
            h_resolution,
            w_resolution
        )

    model.load_state_dict(vis_sd, strict=True)
    return model
