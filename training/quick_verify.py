"""Verify model output shape and bias initialization."""
import torch, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from models.transformer import TransformerDecoder

decoder = TransformerDecoder(64, 2, 6, 256, 3, 50, 0.0, aux_loss=True)
memory = torch.randn(2, 100, 64)
result = decoder(memory)

logits = result['pred_logits']
print(f"Logits shape: {logits.shape}")  # [2, 50, 4] — 3 classes + 1 ∅

# Check bias values
for i, head in enumerate(decoder.cls_head):
    bias = head.bias.detach()
    print(f"  cls_head[{i}] bias: real={bias[:-1].mean().item():.2f}, ∅={bias[-1].item():.2f}")
    print(f"    → real sigmoid: {bias[:-1].sigmoid().mean().item():.4f}, ∅ sigmoid: {bias[-1].sigmoid().item():.4f}")
