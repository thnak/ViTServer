@echo off
cd /d C:\Users\thanh\Git\ViTServer\training
call conda activate base
python train.py --config configs/coco_3cls.yaml --data_path data/coco_mini --device cuda > training_3cls_v2.log 2>&1
