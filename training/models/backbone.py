"""CNN backbone producing P3/P4/P5 multi-scale feature maps.

For 1280×1280 input:
  P3 → [B, 4C, 80, 80]   = 6 400 spatial tokens
  P4 → [B, 8C, 40, 40]   = 1 600 spatial tokens
  P5 → [B, 16C, 20, 20]  =   400 spatial tokens
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor


class ConvBnSiLU(nn.Module):
    def __init__(self, in_c: int, out_c: int, k: int = 3, s: int = 1, p: int = 1) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_c, out_c, k, s, p, bias=False)
        self.bn = nn.BatchNorm2d(out_c, eps=1e-3, momentum=0.03)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x: Tensor) -> Tensor:
        return self.act(self.bn(self.conv(x)))


class Bottleneck(nn.Module):
    def __init__(self, c: int) -> None:
        super().__init__()
        h = c // 2
        self.cv1 = ConvBnSiLU(c, h, 1, p=0)
        self.cv2 = ConvBnSiLU(h, h, 3, p=1)
        self.cv3 = ConvBnSiLU(h, c, 1, p=0)

    def forward(self, x: Tensor) -> Tensor:
        return x + self.cv3(self.cv2(self.cv1(x)))


class C2f(nn.Module):
    """Cross-stage partial with two-branch split + n bottlenecks."""

    def __init__(self, in_c: int, out_c: int, n: int = 1) -> None:
        super().__init__()
        half = out_c // 2
        self.cv1 = ConvBnSiLU(in_c, out_c, 1, p=0)
        self.cv2 = ConvBnSiLU((1 + n) * half, out_c, 1, p=0)
        self.blocks = nn.ModuleList(Bottleneck(half) for _ in range(n))

    def forward(self, x: Tensor) -> Tensor:
        y = list(self.cv1(x).chunk(2, dim=1))
        for b in self.blocks:
            y.append(b(y[-1]))
        return self.cv2(torch.cat(y, dim=1))


class DownConv(nn.Module):
    """Stride-2 conv for spatial downsampling — replaces Slice/PixelShuffle."""

    def __init__(self, in_c: int, out_c: int) -> None:
        super().__init__()
        self.conv = ConvBnSiLU(in_c, out_c, k=2, s=2, p=0)

    def forward(self, x: Tensor) -> Tensor:
        return self.conv(x)


class CNNBackbone(nn.Module):
    """Six-stage CNN backbone — all downsampling via Conv(k=2, s=2), no slice."""

    def __init__(self, in_channels: int = 3, base_channels: int = 64) -> None:
        super().__init__()
        c = base_channels

        # Stem: 1280 → 320  (stride 4 via two consecutive stride-2 convs)
        self.stem = nn.Sequential(
            ConvBnSiLU(in_channels, c, k=2, s=2, p=0),  # 1280 → 640
            ConvBnSiLU(c, c, k=2, s=2, p=0),            # 640  → 320
            C2f(c, c, n=1),
        )
        # Stage 2: 320 → 160
        self.stage2 = nn.Sequential(
            DownConv(c, c * 2),
            C2f(c * 2, c * 2, n=2),
        )
        # Stage 3: 160 → 80  (P3)
        self.stage3 = nn.Sequential(
            DownConv(c * 2, c * 4),
            C2f(c * 4, c * 4, n=2),
        )
        # Stage 4: 80 → 40  (P4)
        self.stage4 = nn.Sequential(
            DownConv(c * 4, c * 8),
            C2f(c * 8, c * 8, n=2),
        )
        # Stage 5: 40 → 20  (P5)
        self.stage5 = nn.Sequential(
            DownConv(c * 8, c * 16),
            C2f(c * 16, c * 16, n=1),
        )

        self.out_channels = (c * 4, c * 8, c * 16)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        x = self.stem(x)
        x = self.stage2(x)
        p3 = self.stage3(x)
        p4 = self.stage4(p3)
        p5 = self.stage5(p4)
        return p3, p4, p5
