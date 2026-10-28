import torch
import torch.nn as nn
import torch.nn.functional as F


def _fft2(x):
    return torch.fft.fft2(x, dim=(-2, -1))


def _ifft2(x):
    return torch.fft.ifft2(x, dim=(-2, -1)).real


def _amp_phase(x, eps=1e-8):
    return torch.abs(x).clamp_min(eps), torch.angle(x)


def _compose(amplitude, phase):
    return amplitude * torch.cos(phase) + 1j * amplitude * torch.sin(phase)


def _base_norm_2d(channels, base):
    if base == "in":
        return nn.InstanceNorm2d(channels, affine=True)
    if base == "bn":
        return nn.BatchNorm2d(channels, affine=True)
    if base == "ln":
        return nn.GroupNorm(1, channels, affine=True)
    raise ValueError(f"Unsupported 2D norm base: {base}")


class FGSCNorm2D(nn.Module):
    """Foreground/background-aware frequency amplitude normalization."""

    def __init__(
        self,
        channels,
        base="in",
        temperature=1e-1,
        constrained=True,
        spatial_split=False,
        min_fg_norm=0.05,
        min_bg_gap=0.10,
        max_norm=0.95,
    ):
        super().__init__()
        self.base = _base_norm_2d(channels, base)
        self.lam_fg = nn.Parameter(torch.tensor([-0.01, 0.01], dtype=torch.float32))
        self.lam_bg = nn.Parameter(torch.tensor([0.01, -0.01], dtype=torch.float32))
        self.temperature = temperature
        self.constrained = bool(constrained)
        self.spatial_split = bool(spatial_split)
        self.min_fg_norm = float(min_fg_norm)
        self.min_bg_gap = float(min_bg_gap)
        self.max_norm = float(max_norm)

    def _norm_ratios(self):
        lam_fg = torch.clamp(self.lam_fg, -10.0, 10.0)
        lam_bg = torch.clamp(self.lam_bg, -10.0, 10.0)
        w_fg = torch.softmax(lam_fg / self.temperature, dim=0)
        w_bg = torch.softmax(lam_bg / self.temperature, dim=0)

        if not self.constrained:
            return w_fg[0], w_bg[0]

        max_norm = min(max(self.max_norm, 1e-3), 0.999)
        min_gap = max(self.min_bg_gap, 0.0)
        min_fg = max(self.min_fg_norm, 0.0)
        max_fg = max_norm - min_gap
        if max_fg <= min_fg:
            min_fg = max(0.0, max_fg - 1e-3)

        fg_norm_ratio = torch.clamp(w_fg[0], min=min_fg, max=max_fg)
        bg_norm_ratio = torch.minimum(
            torch.maximum(w_bg[0], fg_norm_ratio + min_gap),
            w_bg[0].new_tensor(max_norm),
        )
        return fg_norm_ratio, bg_norm_ratio

    def _soft_fg_mask(self, x):
        energy = x.pow(2).mean(dim=1, keepdim=True)
        e_min = energy.amin(dim=(-2, -1), keepdim=True)
        e_max = energy.amax(dim=(-2, -1), keepdim=True)
        return (energy - e_min) / (e_max - e_min + 1e-6)

    def forward(self, x, fg_mask=None):
        x_norm = self.base(x)
        if fg_mask is None:
            fg_mask = self._soft_fg_mask(x)
        fg_mask = fg_mask.clamp(0.0, 1.0)

        fg_norm_ratio, bg_norm_ratio = self._norm_ratios()
        if self.spatial_split:
            return self._forward_spatial_split(x, x_norm, fg_mask, fg_norm_ratio, bg_norm_ratio)

        f_org = _fft2(x)
        f_norm = _fft2(x_norm)
        a_org, p_org = _amp_phase(f_org)
        a_norm, _ = _amp_phase(f_norm)
        a_fg = fg_norm_ratio * a_norm + (1.0 - fg_norm_ratio) * a_org
        a_bg = bg_norm_ratio * a_norm + (1.0 - bg_norm_ratio) * a_org
        a_mix = fg_mask * a_fg + (1.0 - fg_mask) * a_bg
        return _ifft2(_compose(a_mix, p_org))

    def _forward_spatial_split(self, x, x_norm, fg_mask, fg_ratio, bg_ratio):
        bg_mask = 1.0 - fg_mask

        f_fg_org = _fft2(x * fg_mask)
        f_fg_norm = _fft2(x_norm * fg_mask)
        a_fg_org, p_fg_org = _amp_phase(f_fg_org)
        a_fg_norm, _ = _amp_phase(f_fg_norm)
        y_fg = _ifft2(_compose(fg_ratio * a_fg_norm + (1.0 - fg_ratio) * a_fg_org, p_fg_org))

        f_bg_org = _fft2(x * bg_mask)
        f_bg_norm = _fft2(x_norm * bg_mask)
        a_bg_org, p_bg_org = _amp_phase(f_bg_org)
        a_bg_norm, _ = _amp_phase(f_bg_norm)
        y_bg = _ifft2(_compose(bg_ratio * a_bg_norm + (1.0 - bg_ratio) * a_bg_org, p_bg_org))
        return y_fg + y_bg


class SCNorm1D(nn.Module):
    """Frequency amplitude normalization for CLS-token features."""

    def __init__(self, channels, base="ln", Ts=1e-1):
        super().__init__()
        self.base_type = base
        if base == "ln":
            self.base = nn.LayerNorm(channels)
        elif base == "bn":
            self.base = nn.BatchNorm1d(channels, affine=True)
        else:
            raise ValueError(f"Unsupported 1D norm base: {base}")
        self.lam = nn.Parameter(torch.zeros(2))
        self.temperature = Ts

    def forward(self, x):
        if x.dim() != 2:
            raise ValueError(f"SCNorm1D expects (B, C), got {tuple(x.shape)}")
        x_norm = self.base(x)
        f_org = torch.fft.fft(x, dim=-1)
        f_norm = torch.fft.fft(x_norm, dim=-1)
        a_org = torch.abs(f_org).clamp_min(1e-8)
        p_org = torch.angle(f_org)
        a_norm = torch.abs(f_norm).clamp_min(1e-8)
        weights = torch.softmax(torch.clamp(self.lam, -10.0, 10.0) / self.temperature, dim=0)
        a_mix = weights[0] * a_norm + weights[1] * a_org
        return torch.fft.ifft(a_mix * torch.exp(1j * p_org), dim=-1).real


def tokens_to_grid(tokens, height, width):
    batch, num_tokens, channels = tokens.shape
    assert num_tokens == height * width, f"N={num_tokens} should equal H*W={height * width}"
    return tokens.transpose(1, 2).contiguous().view(batch, channels, height, width)


def grid_to_tokens(grid):
    batch, channels, height, width = grid.shape
    return grid.reshape(batch, channels, height * width).transpose(1, 2).contiguous()


class FreqNormOnTokens(nn.Module):
    """Apply SCFG to ViT patch tokens while keeping the CLS token unchanged."""

    def __init__(
        self,
        C,
        H,
        W,
        kind="scfg",
        base="in",
        with_cls=True,
        fg_ratio=0.4,
        fg_mask_temp=0.15,
        scfg_constrained=True,
        scfg_spatial_split=False,
        scfg_min_fg_norm=0.05,
        scfg_min_bg_gap=0.10,
        scfg_max_norm=0.95,
        scfg_warmup_epochs=0,
    ):
        super().__init__()
        if kind != "scfg":
            raise ValueError("The clean MetaN release keeps only kind='scfg'.")
        self.norm = FGSCNorm2D(
            C,
            base=base,
            constrained=scfg_constrained,
            spatial_split=scfg_spatial_split,
            min_fg_norm=scfg_min_fg_norm,
            min_bg_gap=scfg_min_bg_gap,
            max_norm=scfg_max_norm,
        )
        self.H = H
        self.W = W
        self.with_cls = with_cls
        self.fg_ratio = float(fg_ratio)
        self.fg_mask_temp = float(fg_mask_temp)
        self.scfg_warmup_epochs = int(max(0, scfg_warmup_epochs))
        self.register_buffer(
            "_scfg_warmup_scale",
            torch.tensor(1.0, dtype=torch.float32),
            persistent=False,
        )

    def set_epoch(self, epoch):
        if self.scfg_warmup_epochs <= 0:
            self._scfg_warmup_scale.fill_(1.0)
            return
        scale = min(1.0, max(int(epoch), 1) / float(self.scfg_warmup_epochs))
        self._scfg_warmup_scale.fill_(float(scale))

    def _blend(self, original, scfg_output):
        if self.scfg_warmup_epochs <= 0:
            return scfg_output
        scale = float(self._scfg_warmup_scale.item())
        return original + scale * (scfg_output - original)

    def _token_mask_to_grid(self, token_mask, height, width):
        batch, num_tokens, _ = token_mask.shape
        assert num_tokens == height * width
        return token_mask.transpose(1, 2).contiguous().view(batch, 1, height, width)

    def _build_fg_mask(self, cls_token, patch_tokens):
        cls_feat = F.normalize(cls_token.squeeze(1), p=2, dim=1)
        patch_feat = F.normalize(patch_tokens, p=2, dim=2)
        sim = torch.einsum("bnc,bc->bn", patch_feat, cls_feat)
        sim_min = sim.amin(dim=1, keepdim=True)
        sim_max = sim.amax(dim=1, keepdim=True)
        sim_norm = (sim - sim_min) / (sim_max - sim_min + 1e-6)
        q = 1.0 - min(max(self.fg_ratio, 0.05), 0.95)
        threshold = torch.quantile(sim_norm.detach().float(), q=q, dim=1, keepdim=True)
        threshold = threshold.to(sim_norm.dtype)
        temp = max(self.fg_mask_temp, 1e-3)
        return torch.sigmoid((sim_norm - threshold) / temp).unsqueeze(-1)

    def forward(self, tokens):
        if self.with_cls and tokens.size(1) == self.H * self.W + 1:
            cls_token = tokens[:, 0:1, :]
            patch_tokens = tokens[:, 1:, :]
            grid = tokens_to_grid(patch_tokens, self.H, self.W)
            fg_mask = self._token_mask_to_grid(
                self._build_fg_mask(cls_token, patch_tokens),
                self.H,
                self.W,
            )
            out = self.norm(grid, fg_mask=fg_mask)
            out = self._blend(grid, out)
            return torch.cat([cls_token, grid_to_tokens(out)], dim=1)

        num_tokens = tokens.size(1)
        height = width = int(num_tokens**0.5)
        if height * width != num_tokens:
            raise ValueError(f"Cannot infer square grid from {num_tokens} tokens.")
        grid = tokens_to_grid(tokens, height, width)
        out = self.norm(grid)
        return grid_to_tokens(self._blend(grid, out))
