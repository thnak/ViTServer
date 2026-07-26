@echo off
cd /d C:\Users\thanh\Git\ViTServer\training
C:\Users\thanh\miniconda3\Scripts\conda.exe run -n base --no-capture-output python train.py --config configs/coco_3cls.yaml --data_path data/coco_mini --device cuda > training_3cls_v3.log 2>&1
