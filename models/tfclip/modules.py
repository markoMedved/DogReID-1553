import torch
import torch.nn as nn
from collections import OrderedDict
from .clip_model import LayerNorm, QuickGELU


def trunc_normal_(tensor: torch.Tensor, mean: float = 0., std: float = 0.02):
    """Truncated normal initialization."""
    with torch.no_grad():
        return tensor.normal_().fmod_(2).mul_(std).add_(mean)


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample."""
    def __init__(self, drop_prob: float = 0.):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.drop_prob == 0. or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
        if keep_prob > 0.0:
            random_tensor.div_(keep_prob)
        return x * random_tensor


class CrossFramelAttentionBlock(nn.Module):
    """Temporal Cross-Frame Communication Block (TMC + MD) from TF-CLIP."""
    def __init__(self, d_model: int = 768, n_head: int = 12, attn_mask: torch.Tensor = None, droppath: float = 0., T: int = 8):
        super().__init__()
        self.T = T

        # TMC: Temporal Message Construction
        self.message_fc = nn.Linear(d_model, d_model)
        self.message_ln = LayerNorm(d_model)
        self.message_attn = nn.MultiheadAttention(d_model, n_head)

        # MD: Memory Diffusion
        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.drop_path = DropPath(droppath) if droppath > 0. else nn.Identity()
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
        # x shape: [L, BT, D]
        l, bt, d = x.size()
        b = bt // self.T
        x = x.view(l, b, self.T, d)

        # 1. TMC: Extract message token per frame, communicate temporally across T
        msg_token = self.message_fc(x.mean(0))  # [B, T, D]
        msg_token = msg_token.view(b, self.T, 1, d)
        msg_token = msg_token.permute(1, 2, 0, 3).view(self.T, b, d)  # [T, B, D]
        
        msg_token_ln = self.message_ln(msg_token)
        msg_token = msg_token + self.drop_path(
            self.message_attn(msg_token_ln, msg_token_ln, msg_token_ln, need_weights=False)[0]
        )
        msg_token = msg_token.view(self.T, 1, b, d).permute(1, 2, 0, 3)  # [1, B, T, D]

        # 2. MD: Concat message token along spatial dimension and diffuse
        x = torch.cat([x, msg_token], dim=0)  # [L+1, B, T, D]
        x = x.view(l + 1, -1, d)  # [L+1, BT, D]
        x = x + self.drop_path(self.attention(self.ln_1(x)))
        x = x + self.drop_path(self.mlp(self.ln_2(x)))
        return x


class Temporal_Memory_Difusion(nn.Module):
    """TMD Module: Temporal communication across video frames."""
    def __init__(self, width: int = 768, layers: int = 1, heads: int = 12, droppath: list = None, T: int = 8):
        super().__init__()
        if droppath is None:
            droppath = [0.0 for _ in range(layers)]
        self.resblocks = nn.Sequential(*[
            CrossFramelAttentionBlock(width, heads, droppath=droppath[i], T=T) for i in range(layers)
        ])

    def forward(self, x: torch.Tensor):
        return self.resblocks(x)


class MultiHeadAttention(nn.Module):
    """Standard multi-head attention module for SSP."""
    def __init__(self, dim: int, num_heads: int = 8, qkv_bias: bool = False, qk_scale: float = None, attn_drop: float = 0., proj_drop: float = 0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.q_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.k_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.v_proj = nn.Linear(dim, dim, bias=qkv_bias)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
        B, N, C = q.shape
        B, M, C = k.shape
        q = self.q_proj(q).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        k = self.k_proj(k).reshape(B, M, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        v = self.v_proj(v).reshape(B, M, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class PromptGeneratorLayer(nn.Module):
    def __init__(self, d_model: int = 512, nhead: int = 8, dropout: float = 0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, nhead, proj_drop=dropout)
        self.cross_attn = MultiHeadAttention(d_model, nhead, proj_drop=dropout)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

        self.dropout = nn.Dropout(dropout)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            QuickGELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model)
        )

    def forward(self, x: torch.Tensor, visual: torch.Tensor):
        q = k = v = self.norm1(x)
        x = x + self.self_attn(q, k, v)
        q = self.norm2(x)
        x = x + self.cross_attn(q, visual, visual)
        x = x + self.dropout(self.mlp(self.norm3(x)))
        return x


class ImageSpecificPrompt(nn.Module):
    """SSP Module: Adapts visual CLIP memory centroids based on the current video sequence."""
    def __init__(self, layers: int = 2, embed_dim: int = 512, alpha: float = 0.1):
        super().__init__()
        self.memory_proj = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim),
            nn.LayerNorm(embed_dim),
        )
        self.text_proj = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim),
        )
        self.out_proj = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim)
        )
        self.decoder = nn.ModuleList([
            PromptGeneratorLayer(embed_dim, embed_dim // 64) for _ in range(layers)
        ])
        self.alpha = nn.Parameter(torch.ones(embed_dim) * alpha)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, memory: torch.Tensor, visual: torch.Tensor):
        # memory: [B, K, 512]
        # visual: [B, num_tokens, 512]
        visual = self.memory_proj(visual)
        memory = self.text_proj(memory)
        for layer in self.decoder:
            memory = layer(memory, visual)
        return self.out_proj(memory)
