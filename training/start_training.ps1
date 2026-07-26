$logFile = "C:\Users\thanh\Git\ViTServer\training\training_3cls_v3.log"
$python = "C:\Users\thanh\miniconda3\python.exe"
$script = "C:\Users\thanh\Git\ViTServer\training\train.py"

$env:PYTHONIOENCODING = "utf-8"

# Start process completely detached
$p = Start-Process -FilePath $python -ArgumentList "$script --config configs/coco_3cls.yaml --data_path data/coco_mini --device cuda" -WorkingDirectory "C:\Users\thanh\Git\ViTServer\training" -RedirectStandardOutput $logFile -RedirectStandardError $logFile -NoNewWindow -PassThru

Write-Output "Training started with PID: $($p.Id)"
Write-Output "Log: $logFile"
