"""Full NMS-Free detector: Backbone → MFE → Encoder → Decoder."""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

from .backbone import CNNBackbone
from .mfe import MFE
from .transformer import TransformerEncoder, TransformerDecoder


class NMSFreeDetector(nn.Module):
    """
    End-to-end object detector with no NMS post-processing.

    Input  : [B, 3, 1280, 1280]  (NCHW; export uses NHWC for TensorRT)
    Output : dict with keys
               pred_boxes   [B, Q, 4]  — cx,cy,w,h ∈ [0,1]
               pred_logits  [B, Q, C]  — raw class logits
               aux_outputs  (training only)
    """

    def __init__(
        self,
        num_classes: int = 80,
        base_channels: int = 64,
        embed_dim: int = 256,
        num_heads: int = 8,
        num_encoder_layers: int = 1,
        num_decoder_layers: int = 6,
        num_queries: int = 300,
        dropout: float = 0.0,
        aux_loss: bool = True,
    ) -> None:
        super().__init__()
        ffn_dim = embed_dim * 4

        self.backbone = CNNBackbone(in_channels=3, base_channels=base_channels)
        self.mfe = MFE(self.backbone.out_channels, embed_dim)
        self.encoder = TransformerEncoder(embed_dim, num_heads, num_encoder_layers, ffn_dim)
        self.decoder = TransformerDecoder(
            embed_dim, num_heads, num_decoder_layers, ffn_dim,
            num_classes, num_queries, dropout, aux_loss,
        )

    def forward(self, x: Tensor) -> dict[str, list[Tensor] | Tensor]:
        p3, p4, p5 = self.backbone(x)
        memory, _ = self.mfe(p3, p4, p5)      # [B, 8400, D]
        memory = self.encoder(memory)
        return self.decoder(memory)

    # ------------------------------------------------------------------
    # ONNX-friendly inference-only forward (no aux_outputs, sigmoid cls)
    # ------------------------------------------------------------------
    def forward_export(self, x: Tensor) -> tuple[Tensor, Tensor]:
        self.decoder.aux_loss = False
        out = self.forward(x)
        boxes = out["pred_boxes"]                         # [B, Q, 4]
        scores = out["pred_logits"].sigmoid()             # [B, Q, C]
        return boxes, scores
