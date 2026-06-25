"""Transformer encoder-decoder for NMS-Free detection.

Encoder: Scale-Aware Intra-Scale Attention (AIFI) on the finest feature level
         followed by cross-scale feature fusion.
Decoder: 6-layer cross-attention decoder with learnable object queries.
"""

from __future__ import annotations

import copy
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

class MLP(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, num_layers: int) -> None:
        super().__init__()
        dims = [in_dim] + [hidden_dim] * (num_layers - 1) + [out_dim]
        self.layers = nn.ModuleList(
            nn.Linear(dims[i], dims[i + 1]) for i in range(num_layers)
        )

    def forward(self, x: Tensor) -> Tensor:
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < len(self.layers) - 1 else layer(x)
        return x


class TransformerEncoderLayer(nn.Module):
    def __init__(self, d: int, nhead: int, ffn_dim: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.attn = nn.MultiheadAttention(d, nhead, dropout=dropout, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(d, ffn_dim), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(ffn_dim, d), nn.Dropout(dropout),
        )
        self.norm1 = nn.LayerNorm(d)
        self.norm2 = nn.LayerNorm(d)

    def forward(self, x: Tensor) -> Tensor:
        x = self.norm1(x + self.attn(x, x, x)[0])
        return self.norm2(x + self.ffn(x))


class TransformerDecoderLayer(nn.Module):
    def __init__(self, d: int, nhead: int, ffn_dim: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d, nhead, dropout=dropout, batch_first=True)
        self.cross_attn = nn.MultiheadAttention(d, nhead, dropout=dropout, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(d, ffn_dim), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(ffn_dim, d), nn.Dropout(dropout),
        )
        self.norm1 = nn.LayerNorm(d)
        self.norm2 = nn.LayerNorm(d)
        self.norm3 = nn.LayerNorm(d)

    def forward(self, tgt: Tensor, memory: Tensor) -> Tensor:
        tgt = self.norm1(tgt + self.self_attn(tgt, tgt, tgt)[0])
        tgt = self.norm2(tgt + self.cross_attn(tgt, memory, memory)[0])
        return self.norm3(tgt + self.ffn(tgt))


# ---------------------------------------------------------------------------
# Encoder: Intra-Scale Attention on P5, then broadcast to memory
# ---------------------------------------------------------------------------

class TransformerEncoder(nn.Module):
    def __init__(self, d: int, nhead: int, num_layers: int, ffn_dim: int) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            TransformerEncoderLayer(d, nhead, ffn_dim) for _ in range(num_layers)
        )

    def forward(self, x: Tensor) -> Tensor:
        for layer in self.layers:
            x = layer(x)
        return x


# ---------------------------------------------------------------------------
# Decoder: Object Queries × 6 cross-attention layers
# ---------------------------------------------------------------------------

class TransformerDecoder(nn.Module):
    def __init__(
        self,
        d: int,
        nhead: int,
        num_layers: int,
        ffn_dim: int,
        num_classes: int,
        num_queries: int,
        dropout: float = 0.0,
        aux_loss: bool = True,
    ) -> None:
        super().__init__()
        layer = TransformerDecoderLayer(d, nhead, ffn_dim, dropout)
        self.layers = nn.ModuleList(copy.deepcopy(layer) for _ in range(num_layers))

        self.query_embed = nn.Embedding(num_queries, d)

        self.bbox_head = nn.ModuleList(
            MLP(d, d, 4, 3) for _ in range(num_layers)
        )
        self.cls_head = nn.ModuleList(
            nn.Linear(d, num_classes) for _ in range(num_layers)
        )

        self.aux_loss = aux_loss

    def forward(self, memory: Tensor) -> dict[str, list[Tensor]]:
        B = memory.shape[0]
        tgt = self.query_embed.weight.unsqueeze(0).expand(B, -1, -1)

        all_boxes: list[Tensor] = []
        all_logits: list[Tensor] = []

        for i, layer in enumerate(self.layers):
            tgt = layer(tgt, memory)
            boxes = self.bbox_head[i](tgt).sigmoid()    # [B, Q, 4] — cx,cy,w,h ∈ [0,1]
            logits = self.cls_head[i](tgt)              # [B, Q, num_classes]
            all_boxes.append(boxes)
            all_logits.append(logits)

        out = {"pred_boxes": all_boxes[-1], "pred_logits": all_logits[-1]}
        if self.aux_loss:
            out["aux_outputs"] = [
                {"pred_boxes": b, "pred_logits": l}
                for b, l in zip(all_boxes[:-1], all_logits[:-1])
            ]
        return out
