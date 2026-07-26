"""Verify the no-object loss logic is correct."""
import torch, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from losses.hungarian import HungarianMatcher, HungarianCriterion

B, Q, C = 2, 50, 3  # 3 real classes

matcher = HungarianMatcher(2.0, 5.0, 2.0)
criterion = HungarianCriterion(num_classes=C, matcher=matcher, no_obj_weight=0.5)

logits = torch.randn(B, Q, C+1)  # C+1 includes ∅
boxes = torch.rand(B, Q, 4) * 0.5 + 0.25

targets = []
for b in range(B):
    n = 2 if b == 0 else 1
    labels = torch.zeros(100, dtype=torch.long)
    labels[:n] = torch.randint(0, C, (n,))
    t_boxes = torch.zeros(100, 4)
    t_boxes[:n] = torch.rand(n, 4) * 0.3 + 0.35
    valid = torch.zeros(100, dtype=torch.bool)
    valid[:n] = True
    targets.append({'labels': labels, 'boxes': t_boxes, 'valid': valid,
                    'image_id': torch.tensor(1), 'orig_size': torch.tensor([416, 416])})

outputs = {'pred_logits': logits, 'pred_boxes': boxes}
losses = criterion(outputs, targets)
print('Loss keys:', list(losses.keys()))
for k, v in losses.items():
    print(f'  {k}: {v.item():.4f}')
print('Total:', losses['total'].item())
print('Test PASSED')
