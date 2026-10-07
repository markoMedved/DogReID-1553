"""Native ResNet-50-IBN-a backbone implementation in pure PyTorch.

Supports:
- IBN-a (Instance-Batch Normalization): IN on half channels, BN on half channels in layer1, layer2, layer3.
- last_stride: stride=1 in layer4 (preserving spatial resolution for fine-grained re-ID).
- Non-Local neural network blocks (for AGW and SBS).
- Pretrained weights auto-loading from torch cache (~/.cache/torch/hub/checkpoints/resnet50_ibn_a-d9d0bb7b.pth).
"""

import math
import os
import torch
import torch.nn as nn


class IBN(nn.Module):
    """IBN-a module: splits channels in half, applying InstanceNorm to the first

    half and BatchNorm to the second half.
    Pan et al., 'Two at Once: Enhancing Learning and Generalization Capacities via IBN-Net', ECCV 2018.
    """

    def __init__(self, planes: int):
        super().__init__()
        half1 = planes // 2
        half2 = planes - half1
        self.half = half1
        self.IN = nn.InstanceNorm2d(half1, affine=True)
        self.BN = nn.BatchNorm2d(half2, eps=1e-5, momentum=0.1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        split = torch.split(x, self.half, dim=1)
        out1 = self.IN(split[0].contiguous())
        out2 = self.BN(split[1].contiguous())
        return torch.cat((out1, out2), dim=1)


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: nn.Module = None,
        with_ibn: bool = False
    ):
        super().__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = IBN(planes) if with_ibn else nn.BatchNorm2d(planes, eps=1e-5, momentum=0.1)

        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes, eps=1e-5, momentum=0.1)

        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion, eps=1e-5, momentum=0.1)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)
        return out


class NonLocal2d(nn.Module):
    """Non-Local Neural Network block from Wang et al., CVPR 2018.

    Matches OpenAnimals / FastReID AGW architecture.
    W normalization layer is initialized with zeros so the block is an exact identity mapping at initialization.
    """

    def __init__(self, in_channels: int, reduc_ratio: int = 2):
        super().__init__()
        self.in_channels = in_channels
        # Matches OpenAnimals layer configuration
        self.inter_channels = max(1, reduc_ratio // reduc_ratio)

        self.g = nn.Conv2d(in_channels, self.inter_channels, kernel_size=1, bias=False)
        self.theta = nn.Conv2d(in_channels, self.inter_channels, kernel_size=1, bias=False)
        self.phi = nn.Conv2d(in_channels, self.inter_channels, kernel_size=1, bias=False)

        self.W = nn.Sequential(
            nn.Conv2d(self.inter_channels, in_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(in_channels, eps=1e-5, momentum=0.1)
        )
        nn.init.constant_(self.W[1].weight, 0.0)
        nn.init.constant_(self.W[1].bias, 0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.size(0)
        g_x = self.g(x).view(batch_size, self.inter_channels, -1).permute(0, 2, 1)

        theta_x = self.theta(x).view(batch_size, self.inter_channels, -1).permute(0, 2, 1)
        phi_x = self.phi(x).view(batch_size, self.inter_channels, -1)

        f = torch.matmul(theta_x, phi_x)
        N = f.size(-1)
        f_div_C = f / N

        y = torch.matmul(f_div_C, g_x)
        y = y.permute(0, 2, 1).contiguous()
        y = y.view(batch_size, self.inter_channels, *x.size()[2:])

        return self.W(y) + x


class ResNet50_IBN_a(nn.Module):
    """Native ResNet-50-IBN-a backbone.

    Args:
        last_stride: Stride of layer4 downsampling (default: 1 for Re-ID).
        with_nl: Whether to insert Non-Local blocks in layer2 and layer3 (for AGW & SBS).
        pretrained: Whether to load pre-trained ImageNet weights.
        pretrained_path: Explicit path to checkpoint file.
    """

    def __init__(
        self,
        last_stride: int = 1,
        with_nl: bool = False,
        pretrained: bool = True,
        pretrained_path: str = None
    ):
        super().__init__()
        self.inplanes = 64
        self.dim = 2048

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64, eps=1e-5, momentum=0.1)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True)

        self.layer1 = self._make_layer(Bottleneck, 64, blocks=3, stride=1, with_ibn=True)
        self.layer2 = self._make_layer(Bottleneck, 128, blocks=4, stride=2, with_ibn=True)
        self.layer3 = self._make_layer(Bottleneck, 256, blocks=6, stride=2, with_ibn=True)
        self.layer4 = self._make_layer(Bottleneck, 512, blocks=3, stride=last_stride, with_ibn=False)

        self.with_nl = with_nl
        if with_nl:
            # Matches OpenAnimals AGW / SBS: 2 NL blocks in stage 2 (indices 2, 3) and 3 in stage 3 (indices 3, 4, 5)
            self.NL_2 = nn.ModuleList([NonLocal2d(512) for _ in range(2)])
            self.NL_2_idx = [2, 3]
            self.NL_3 = nn.ModuleList([NonLocal2d(1024) for _ in range(3)])
            self.NL_3_idx = [3, 4, 5]
        else:
            self.NL_2_idx = []
            self.NL_3_idx = []

        self._init_weights()

        if pretrained:
            self.load_pretrained_weights(pretrained_path)

    def _make_layer(
        self,
        block,
        planes: int,
        blocks: int,
        stride: int = 1,
        with_ibn: bool = False
    ) -> nn.Sequential:
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion, eps=1e-5, momentum=0.1)
            )

        layers = [block(self.inplanes, planes, stride=stride, downsample=downsample, with_ibn=with_ibn)]
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, with_ibn=with_ibn))

        return nn.Sequential(*layers)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.InstanceNorm2d)):
                if m.weight is not None:
                    nn.init.constant_(m.weight, 1.0)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)

        # Layer 2 with optional Non-Local blocks
        if self.with_nl and len(self.NL_2_idx) > 0:
            nl_cnt = 0
            for i, blk in enumerate(self.layer2):
                x = blk(x)
                if nl_cnt < len(self.NL_2_idx) and i == self.NL_2_idx[nl_cnt]:
                    x = self.NL_2[nl_cnt](x)
                    nl_cnt += 1
        else:
            x = self.layer2(x)

        # Layer 3 with optional Non-Local blocks
        if self.with_nl and len(self.NL_3_idx) > 0:
            nl_cnt = 0
            for i, blk in enumerate(self.layer3):
                x = blk(x)
                if nl_cnt < len(self.NL_3_idx) and i == self.NL_3_idx[nl_cnt]:
                    x = self.NL_3[nl_cnt](x)
                    nl_cnt += 1
        else:
            x = self.layer3(x)

        x = self.layer4(x)
        return x

    def load_pretrained_weights(self, path: str = None):
        """Load pretrained ResNet-50-IBN-a weights."""
        if path is None:
            default_path = os.path.expanduser("~/.cache/torch/hub/checkpoints/resnet50_ibn_a-d9d0bb7b.pth")
            if os.path.exists(default_path):
                path = default_path
            else:
                url = "https://github.com/XingangPan/IBN-Net/releases/download/v1.0/resnet50_ibn_a-d9d0bb7b.pth"
                os.makedirs(os.path.dirname(default_path), exist_ok=True)
                print(f"[resnet50_ibn_a] Downloading pretrained weights from {url} to {default_path}...")
                torch.hub.download_url_to_file(url, default_path)
                path = default_path

        state_dict = torch.load(path, map_location="cpu")
        # Strip fc classifier keys if present
        filtered_state = {k: v for k, v in state_dict.items() if not k.startswith("fc.")}
        missing, unexpected = self.load_state_dict(filtered_state, strict=False)
        expected_missing = [k for k in missing if k.startswith("NL_")]
        other_missing = [k for k in missing if not k.startswith("NL_")]
        print(f"[resnet50_ibn_a] Loaded pretrained weights from {path} (missing: {len(missing)} [NL: {len(expected_missing)}], unexpected: {len(unexpected)})")
        if other_missing:
            print(f"[resnet50_ibn_a] Warning: unexpected missing keys: {other_missing[:5]}")
