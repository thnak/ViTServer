"""Check v3 model with no-object class — do queries predict different classes?"""
import torch, yaml, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from models import NMSFreeDetector
from datasets import build_dataloader
from losses.bbox_loss import box_cxcywh_to_xyxy, box_iou

device = torch.device('cuda')
with open('configs/coco_3cls.yaml') as f:
    cfg = yaml.safe_load(f)
mc = cfg['model']

model = NMSFreeDetector(
    num_classes=mc['num_classes'], base_channels=mc['base_channels'],
    embed_dim=mc['embed_dim'], num_heads=mc['num_heads'],
    num_encoder_layers=mc['num_encoder_layers'], num_decoder_layers=mc['num_decoder_layers'],
    num_queries=mc['num_queries'], dropout=mc['dropout'], aux_loss=mc['aux_loss'],
).to(device)
ckpt = torch.load('runs/coco_3cls_v2/last.pt', map_location=device)
model.load_state_dict(ckpt['model'])
model.eval()
print(f'Loaded epoch {ckpt["epoch"]}')

dc = cfg['data']
data_root = Path('data/coco_mini')
loader = build_dataloader(
    str(data_root / dc['train_img_dir']), str(data_root / dc['train_ann']),
    img_size=mc['img_size'], batch_size=4, num_workers=0, train=False, pin_memory=False)

images, targets = next(iter(loader))
images = images.to(device)
with torch.no_grad():
    out = model(images)

logits = out['pred_logits']  # [B, Q, 4] — C+1 (3 real + 1 ∅)
no_obj = logits[:, :, -1].sigmoid()  # ∅ probability
real_scores = logits[:, :, :-1].sigmoid()  # [B, Q, 3]

for b in range(4):
    gt_valid = targets[b]['valid']
    gt_boxes = targets[b]['boxes'][gt_valid].to(device)
    gt_labels = targets[b]['labels'][gt_valid]
    n_gt = len(gt_boxes)

    pred_b = out['pred_boxes'][b]
    no_obj_b = no_obj[b]
    real_b = real_scores[b]
    max_real, pred_cls = real_b.max(dim=1)

    # How many predict no-object (∅)?
    is_noobj = no_obj_b > 0.5
    print(f'\nImg {b}: {n_gt} GT, no-obj={is_noobj.sum().item()}/50 (∅ score mean={no_obj_b.mean():.3f})')

    # Class distribution among non-∅ predictions
    real_q = (~is_noobj).nonzero(as_tuple=True)[0]
    if len(real_q) > 0:
        cls_counts = {c: (pred_cls[real_q] == c).sum().item() for c in range(3)}
        print(f'  Non-∅ class dist: {cls_counts}, mean_score={max_real[real_q].mean():.3f} max_score={max_real[real_q].max():.3f}')
    else:
        print(f'  All queries predict ∅')

    # Best IoU for each GT
    for gi in range(n_gt):
        gt = gt_boxes[gi:gi+1]
        p_xyxy = box_cxcywh_to_xyxy(pred_b)
        t_xyxy = box_cxcywh_to_xyxy(gt.expand(50, -1))
        iou, _ = box_iou(p_xyxy, t_xyxy)
        best_iou, best_idx = iou.max(0)
        pred_cls_i = pred_cls[best_idx].item()
        gt_cls = int(gt_labels[gi].item())
        print(f'  GT{gi}: cls={gt_cls} bestIoU={best_iou.item():.3f} pred_cls={pred_cls_i} {"✓" if pred_cls_i==gt_cls else "✗"}')
