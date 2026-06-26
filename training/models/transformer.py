"""Transformer encoder-decoder for NMS-Free detection.

Encoder options (set via NMSFreeDetector encoder_type):
  "none"   — no encoder; MFE tokens flow directly to decoder (default).
             Decoder cross-attention handles all inter-scale reasoning.
  "window" — per-scale windowed self-attention (ws×ws windows, O(N×ws²)).
             Sees all three scales; no O(N²) full-sequence bottleneck.
  "full"   — legacy full self-attention on all 8400 tokens (O(N²)).
             Kept for ablation; not recommended at 640+ px.

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

class MultiheadAttentionONNX(nn.Module):
    """Drop-in for nn.MultiheadAttention that exports cleanly to ONNX opset ≥ 17.

    PyTorch 2.9+ fuses MHA into aten::_native_multi_head_attention which has no
    ONNX symbolic.  This module decomposes to torch.bmm + F.softmax — both have
    been ONNX ops since opset 1.

    Interface matches nn.MultiheadAttention(batch_first=True):
        forward(q, k, v) → (output, None)
    """

    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.0) -> None:
        super().__init__()
        assert embed_dim % num_heads == 0
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim  = embed_dim // num_heads
        self.scale     = self.head_dim ** -0.5
        self.dropout   = dropout
        self.q_proj  = nn.Linear(embed_dim, embed_dim)
        self.k_proj  = nn.Linear(embed_dim, embed_dim)
        self.v_proj  = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, q: Tensor, k: Tensor, v: Tensor) -> tuple[Tensor, None]:
        B, Sq, D = q.shape
        Sk = k.shape[1]
        H, dh = self.num_heads, self.head_dim

        q = self.q_proj(q).view(B, Sq, H, dh).permute(0, 2, 1, 3).reshape(B * H, Sq, dh)
        k = self.k_proj(k).view(B, Sk, H, dh).permute(0, 2, 1, 3).reshape(B * H, Sk, dh)
        v = self.v_proj(v).view(B, Sk, H, dh).permute(0, 2, 1, 3).reshape(B * H, Sk, dh)

        attn = torch.bmm(q, k.transpose(1, 2)) * self.scale   # [B*H, Sq, Sk]
        attn = F.softmax(attn, dim=-1)
        if self.training and self.dropout > 0:
            attn = F.dropout(attn, p=self.dropout)

        out = torch.bmm(attn, v)                               # [B*H, Sq, dh]
        out = out.reshape(B, H, Sq, dh).permute(0, 2, 1, 3).reshape(B, Sq, D)
        return self.out_proj(out), None


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

    def forward(self, x: Tensor, shapes=None) -> Tensor:  # shapes unused (full attention)
        for layer in self.layers:
            x = layer(x)
        return x


# ---------------------------------------------------------------------------
# Window helpers — all contiguous reshape+permute, no gather/scatter
# ---------------------------------------------------------------------------

def _window_partition(x: Tensor, ws: int) -> Tensor:
    """[B, H, W, D] → [B*nW, ws*ws, D].  H and W must be multiples of ws."""
    B, H, W, D = x.shape
    x = x.view(B, H // ws, ws, W // ws, ws, D)
    return x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, ws * ws, D)


def _window_unpartition(windows: Tensor, ws: int, H: int, W: int, B: int) -> Tensor:
    """[B*nW, ws*ws, D] → [B, H, W, D]."""
    D = windows.shape[-1]
    x = windows.view(B, H // ws, W // ws, ws, ws, D)
    return x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, D)


# ---------------------------------------------------------------------------
# Window encoder — per-scale windowed self-attention
# ---------------------------------------------------------------------------

class ScaleWindowEncoderLayer(nn.Module):
    """One layer of per-scale windowed self-attention.

    Each scale's tokens are windowed independently (ws×ws windows).
    Scales that are smaller than a single window get full attention.
    Padding is applied when H or W is not a multiple of ws; unpadded after.
    """

    def __init__(self, d: int, nhead: int, ffn_dim: int,
                 window_size: int = 8, dropout: float = 0.0) -> None:
        super().__init__()
        self.window_size = window_size
        self.attn = nn.MultiheadAttention(d, nhead, dropout=dropout, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(d, ffn_dim), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(ffn_dim, d), nn.Dropout(dropout),
        )
        self.norm1 = nn.LayerNorm(d)
        self.norm2 = nn.LayerNorm(d)

    def _attn_scale(self, tokens: Tensor, h: int, w: int) -> Tensor:
        B, N, D = tokens.shape
        ws = self.window_size
        if h <= ws and w <= ws:
            out, _ = self.attn(tokens, tokens, tokens)
            return out
        # Pad spatial dims to multiples of ws
        pad_h = (ws - h % ws) % ws
        pad_w = (ws - w % ws) % ws
        x = tokens.view(B, h, w, D)
        if pad_h or pad_w:
            x = x.permute(0, 3, 1, 2)                          # [B, D, H, W]
            x = F.pad(x, (0, pad_w, 0, pad_h))
            x = x.permute(0, 2, 3, 1)                          # [B, Hp, Wp, D]
        hp, wp = h + pad_h, w + pad_w
        windows = _window_partition(x, ws)                      # [B*nW, ws², D]
        out_w, _ = self.attn(windows, windows, windows)
        out_x = _window_unpartition(out_w, ws, hp, wp, B)      # [B, Hp, Wp, D]
        if pad_h or pad_w:
            out_x = out_x[:, :h, :w, :].contiguous()
        return out_x.view(B, N, D)

    def forward(self, x: Tensor, shapes: list[tuple[int, int]]) -> Tensor:
        splits = [h * w for h, w in shapes]
        per_scale = list(torch.split(x, splits, dim=1))
        results = []
        for tokens, (h, w) in zip(per_scale, shapes):
            tokens = tokens + self._attn_scale(self.norm1(tokens), h, w)
            tokens = tokens + self.ffn(self.norm2(tokens))
            results.append(tokens)
        return torch.cat(results, dim=1)


class ScaleWindowEncoder(nn.Module):
    """Stack of ScaleWindowEncoderLayers.

    Complexity: O(N × ws²) per layer instead of O(N²).
    At 640 px with ws=8: ~672K ops vs 70M for full attention (104× cheaper).
    Sees all three feature scales (P3/P4/P5) unlike AIFI which only sees P5.
    """

    def __init__(self, d: int, nhead: int, num_layers: int,
                 ffn_dim: int, window_size: int = 8) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            ScaleWindowEncoderLayer(d, nhead, ffn_dim, window_size)
            for _ in range(num_layers)
        )

    def forward(self, x: Tensor, shapes: list[tuple[int, int]]) -> Tensor:
        for layer in self.layers:
            x = layer(x, shapes)
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
