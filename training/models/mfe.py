"""Multi-Scale Feature Embedding (MFE).

Projects P3/P4/P5 CNN features into a common embedding space,
adds 2-D sine positional encoding, then concatenates into a flat
token sequence of length 8 400 (6400 + 1600 + 400).
"""

from __future__ import annotations

import math
import torch
import torch.nn as nn
from torch import Tensor


class SinePositionEncoding2D(nn.Module):
    """Standard 2-D sine/cosine positional encoding."""

    def __init__(self, embed_dim: int, temperature: float = 10_000.0) -> None:
        super().__init__()
        assert embed_dim % 2 == 0
        self.embed_dim = embed_dim
        self.temperature = temperature

    def forward(self, h: int, w: int, device: torch.device) -> Tensor:
        y = torch.arange(h, device=device, dtype=torch.float32)
        x = torch.arange(w, device=device, dtype=torch.float32)
        half = self.embed_dim // 2
        dim_t = torch.arange(half, device=device, dtype=torch.float32)
        dim_t = self.temperature ** (2 * (dim_t // 2) / half)

        pos_x = x.unsqueeze(1) / dim_t          # [W, D/2]
        pos_y = y.unsqueeze(1) / dim_t          # [H, D/2]
        pos_x = torch.stack((pos_x[:, 0::2].sin(), pos_x[:, 1::2].cos()), dim=2).flatten(1)
        pos_y = torch.stack((pos_y[:, 0::2].sin(), pos_y[:, 1::2].cos()), dim=2).flatten(1)

        # [H, W, D]
        pe = torch.cat([
            pos_y.unsqueeze(1).expand(-1, w, -1),
            pos_x.unsqueeze(0).expand(h, -1, -1),
        ], dim=2)
        return pe.flatten(0, 1)  # [H*W, D]


class ScaleProjection(nn.Module):
    """Project one CNN feature map to embed_dim tokens."""

    def __init__(self, in_channels: int, embed_dim: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels, embed_dim, 1, bias=False)
        self.norm = nn.LayerNorm(embed_dim)  # applied on token dim, works at any spatial size

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv(x)                        # [B, D, H, W]
        b, d, h, w = x.shape
        x = x.flatten(2).transpose(1, 2)        # [B, H*W, D]
        return self.norm(x), h, w


class MFE(nn.Module):
    """Multi-Scale Feature Embedding — 8 400 tokens from P3/P4/P5."""

    def __init__(
        self,
        in_channels: tuple[int, int, int],
        embed_dim: int,
    ) -> None:
        super().__init__()
        self.projs = nn.ModuleList(
            ScaleProjection(c, embed_dim) for c in in_channels
        )
        self.pos_enc = SinePositionEncoding2D(embed_dim)
        self.embed_dim = embed_dim
        # Per-scale level embeddings
        self.level_embed = nn.Parameter(torch.zeros(3, embed_dim))
        nn.init.normal_(self.level_embed)

    def forward(
        self, p3: Tensor, p4: Tensor, p5: Tensor
    ) -> tuple[Tensor, list[tuple[int, int]]]:
        features = [p3, p4, p5]
        tokens_list: list[Tensor] = []
        shapes: list[tuple[int, int]] = []

        for i, (proj, feat) in enumerate(zip(self.projs, features)):
            tokens, h, w = proj(feat)       # [B, HW, D]
            pe = self.pos_enc(h, w, feat.device)  # [HW, D]
            tokens = tokens + pe.unsqueeze(0) + self.level_embed[i]
            tokens_list.append(tokens)
            shapes.append((h, w))

        return torch.cat(tokens_list, dim=1), shapes  # [B, 8400, D]
