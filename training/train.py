#!/usr/bin/env python3
"""Training entry point for the NMS-Free detector."""

from __future__ import annotations

import argparse
import contextlib
import math
from copy import deepcopy
from pathlib import Path
from typing import Callable, Optional

import torch
import torch.nn as nn
from torch.amp import GradScaler
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import yaml

from models import NMSFreeDetector
from losses import HungarianCriterion
from losses.hungarian import HungarianMatcher
from datasets import build_dataloader
from utils import MeanAveragePrecision


def _resolve_device(device_arg: str) -> tuple[torch.device, Optional[Callable]]:
    """Return (device, mark_step_fn).

    mark_step_fn is xm.mark_step for XLA/TPU (must be called after each
    optimizer step to flush the lazy graph), None for all other devices.
    """
    if device_arg == "tpu":
        try:
            import torch_xla
            return torch_xla.device(), torch_xla.sync
        except ImportError:
            raise SystemExit(
                "torch_xla is required for TPU training.\n"
                "  pip install torch_xla[tpu] "
                "-f https://storage.googleapis.com/libtpu-releases/index.html"
            )
    if device_arg == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        return torch.device("cpu"), None
    return torch.device(device_arg), None


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("ViTServer NMS-Free Detector Trainer")
    p.add_argument("--config", default="configs/custom_model.yaml")
    p.add_argument(
        "--data_path",
        default="data/coco",
        help="Root of COCO dataset (default: data/coco). "
             "Run scripts/download_coco.py first if not present.",
    )
    p.add_argument("--img_size", type=int, default=None, help="Override img_size from config")
    p.add_argument("--batch_size", type=int, default=None, help="Override batch_size from config")
    p.add_argument("--resume", default="")
    p.add_argument("--device", default="cuda")
    p.add_argument("--no-val", action="store_true", help="Skip validation (faster iteration)")
    p.add_argument("--compile", action="store_true", help="torch.compile the model (CUDA only, ~10-30%% faster)")
    p.add_argument("--compile-mode", default="reduce-overhead",
                   choices=["default", "reduce-overhead", "max-autotune"],
                   help="torch.compile mode (default: reduce-overhead)")
    return p.parse_args()


def build_model_and_criterion(cfg: dict, device: torch.device):
    model_cfg = cfg["model"]
    model = NMSFreeDetector(
        num_classes=model_cfg["num_classes"],
        base_channels=model_cfg["base_channels"],
        embed_dim=model_cfg["embed_dim"],
        num_heads=model_cfg["num_heads"],
        num_encoder_layers=model_cfg["num_encoder_layers"],
        num_decoder_layers=model_cfg["num_decoder_layers"],
        num_queries=model_cfg["num_queries"],
        dropout=model_cfg["dropout"],
        aux_loss=model_cfg["aux_loss"],
        encoder_type=model_cfg.get("encoder_type", "none"),
        window_size=model_cfg.get("window_size", 8),
    ).to(device)

    lc = cfg["loss"]
    matcher = HungarianMatcher(lc["matcher_cls_weight"], lc["matcher_bbox_weight"], lc["matcher_giou_weight"])
    criterion = HungarianCriterion(
        num_classes=model_cfg["num_classes"],
        matcher=matcher,
        cls_weight=lc["cls_weight"],
        bbox_weight=lc["bbox_weight"],
        giou_weight=lc["giou_weight"],
        focal_alpha=lc["focal_alpha"],
        focal_gamma=lc["focal_gamma"],
    ).to(device)

    return model, criterion


def train_one_epoch(
    model: nn.Module,
    criterion: nn.Module,
    loader,
    optimizer: torch.optim.Optimizer,
    scaler: Optional[GradScaler],
    device: torch.device,
    epoch: int,
    amp_enabled: bool = True,
    grad_accum: int = 1,
    clip_norm: float = 0.1,
    mark_step_fn: Optional[Callable] = None,
) -> dict[str, float]:
    model.train()
    criterion.train()
    totals: dict[str, float] = {}
    optimizer.zero_grad()

    pbar = tqdm(loader, desc=f"Epoch {epoch}")
    for step, (images, targets) in enumerate(pbar):
        images = images.to(device, non_blocking=True)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        ac = torch.autocast(device.type, enabled=amp_enabled) if amp_enabled else contextlib.nullcontext()
        with ac:
            outputs = model(images)
            losses = criterion(outputs, targets)

        loss = losses["total"] / grad_accum
        if scaler is not None:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        if (step + 1) % grad_accum == 0:
            if scaler is not None:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), clip_norm)
                scaler.step(optimizer)
                scaler.update()
            else:
                nn.utils.clip_grad_norm_(model.parameters(), clip_norm)
                optimizer.step()
            optimizer.zero_grad()
            if mark_step_fn is not None:
                mark_step_fn()

        for k, v in losses.items():
            totals[k] = totals.get(k, 0.0) + v.item()

        pbar.set_postfix({k: f"{v / (step + 1):.4f}" for k, v in totals.items()})

    n = len(loader)
    return {k: v / n for k, v in totals.items()}


@torch.no_grad()
def validate(
    model: nn.Module,
    loader,
    device: torch.device,
    ann_file: str,
    amp_enabled: bool = True,
) -> dict[str, float]:
    model.eval()
    metric = MeanAveragePrecision(ann_file)
    for images, targets in tqdm(loader, desc="Val"):
        images = images.to(device, non_blocking=True)
        ac = torch.autocast(device.type, enabled=amp_enabled) if amp_enabled else contextlib.nullcontext()
        with ac:
            out = model(images)
        scores = out["pred_logits"].sigmoid()
        ids = [t["image_id"].item() for t in targets]
        orig = torch.stack([t["orig_size"] for t in targets])
        metric.update(out["pred_boxes"].cpu(), scores.cpu(), ids, orig)
    return metric.compute()


class EMA:
    def __init__(self, model: nn.Module, decay: float = 0.9999) -> None:
        self.ema = deepcopy(model).eval()
        self.decay = decay

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        for ema_p, p in zip(self.ema.parameters(), model.parameters()):
            ema_p.mul_(self.decay).add_(p.data, alpha=1 - self.decay)


# ── console helpers ──────────────────────────────────────────────────────────

_EW = (11, 11, 11, 11, 11, 11)                                      # col widths
_EH = ("Epoch", "loss_cls", "loss_l1", "loss_ciou", "total", "lr")  # headers
_HDREP = 20                                                          # reprint every N epochs


def _hline(widths, l="├", m="┼", r="┤") -> str:
    return l + m.join("─" * w for w in widths) + r


def _cells(vals, widths) -> str:
    return "│" + "│".join(f" {str(v):>{w - 2}} " for v, w in zip(vals, widths)) + "│"


def print_run_header(
    cfg: dict, args: argparse.Namespace, model: nn.Module, device: torch.device
) -> None:
    mc, tc = cfg["model"], cfg["training"]
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    compile_str = args.compile_mode if args.compile else "off"

    left = [
        ("img_size",       str(mc["img_size"])),
        ("base_channels",  str(mc["base_channels"])),
        ("embed_dim",      str(mc["embed_dim"])),
        ("num_heads",      str(mc["num_heads"])),
        ("num_queries",    str(mc["num_queries"])),
        ("num_classes",    str(mc["num_classes"])),
        ("encoder_type",   mc.get("encoder_type", "none")),
        ("decoder_layers", str(mc["num_decoder_layers"])),
        ("parameters",     f"{n_params:.2f} M"),
    ]
    right = [
        ("epochs",         str(tc["epochs"])),
        ("batch_size",     str(tc["batch_size"])),
        ("eff_batch",      str(tc["batch_size"] * tc["grad_accumulate"])),
        ("lr",             f"{tc['lr']:.2e}"),
        ("lr_backbone",    f"{tc['lr_backbone']:.2e}"),
        ("amp",            str(tc["amp"])),
        ("ema",            str(tc["ema"])),
        ("warmup",         f"{tc['warmup_epochs']} ep"),
        ("compile",        compile_str),
    ]
    while len(left) < len(right): left.append(("", ""))
    while len(right) < len(left): right.append(("", ""))

    KW, VW = 14, 10
    CW = KW + VW + 6   # "  {key:<14}: {val:>10}  " = 2+14+2+10+2 = 30
    W  = CW * 2 + 3    # │ left(30) │ right(30) │

    title = " ViTServer — NMS-Free Detector "
    print("┌" + "─" * (W - 2) + "┐")
    print("│" + title.center(W - 2) + "│")
    print("├" + "─" * CW + "┬" + "─" * CW + "┤")
    print("│" + " Model".ljust(CW) + "│" + " Training".ljust(CW) + "│")
    print("├" + "─" * CW + "┼" + "─" * CW + "┤")
    for (lk, lv), (rk, rv) in zip(left, right):
        lc = f"  {lk:<{KW}}: {lv:>{VW}}  "
        rc = f"  {rk:<{KW}}: {rv:>{VW}}  "
        print("│" + lc + "│" + rc + "│")
    print("└" + "─" * CW + "┴" + "─" * CW + "┘")


def print_epoch_header() -> None:
    print(_hline(_EW, "┌", "┬", "┐"))
    print(_cells(_EH, _EW))
    print(_hline(_EW))


def print_epoch_row(epoch: int, total: int, metrics: dict, lr: float) -> None:
    vals = (
        f"{epoch + 1}/{total}",
        f"{metrics.get('loss_cls',  0.):.4f}",
        f"{metrics.get('loss_l1',   0.):.4f}",
        f"{metrics.get('loss_ciou', 0.):.4f}",
        f"{metrics.get('total',     0.):.4f}",
        f"{lr:.2e}",
    )
    print(_cells(vals, _EW))


def print_val_metrics(epoch: int, total: int, metrics: dict) -> None:
    pairs = [
        ("mAP",    metrics.get("mAP",   0.), "mAP@50", metrics.get("mAP50", 0.)),
        ("mAP@75", metrics.get("mAP75", 0.), "mAP_s",  metrics.get("mAP_s", 0.)),
        ("mAP_m",  metrics.get("mAP_m", 0.), "mAP_l",  metrics.get("mAP_l", 0.)),
    ]
    lines = [f"  {k1:<8} {v1:.4f}    {k2:<8} {v2:.4f}  " for k1, v1, k2, v2 in pairs]
    inner = max(len(ln) for ln in lines)
    title = f" Val · Epoch {epoch + 1}/{total} "
    print("┌─" + title + "─" * (inner - 1 - len(title)) + "┐")
    for line in lines:
        print("│" + line + "│")
    print("└" + "─" * inner + "┘")


def main() -> None:
    args = parse_args()
    device, mark_step_fn = _resolve_device(args.device)

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    dc = cfg["data"]
    tc = cfg["training"]
    lc = cfg["logging"]

    if args.batch_size is not None:
        tc["batch_size"] = args.batch_size
    if args.img_size is not None:
        cfg["model"]["img_size"] = args.img_size

    data_root = Path(args.data_path)
    if not data_root.exists():
        raise SystemExit(
            f"Dataset not found at '{data_root}'.\n"
            "Run:  python scripts/download_coco.py --dest " + str(data_root)
        )

    pin_memory = dc.get("pin_memory", True) and device.type not in ("cpu", "xla")
    # XLA (TPU) forks dataloader workers which deadlock against XLA's internal state.
    num_workers = 0 if device.type == "xla" else dc["num_workers"]
    train_loader = build_dataloader(
        str(data_root / dc["train_img_dir"]),
        str(data_root / dc["train_ann"]),
        img_size=cfg["model"]["img_size"],
        batch_size=tc["batch_size"],
        num_workers=num_workers,
        train=True,
        pin_memory=pin_memory,
    )
    val_loader = None
    if not args.no_val:
        val_loader = build_dataloader(
            str(data_root / dc["val_img_dir"]),
            str(data_root / dc["val_ann"]),
            img_size=cfg["model"]["img_size"],
            batch_size=tc["batch_size"],
            num_workers=num_workers,
            train=False,
            pin_memory=pin_memory,
        )

    model, criterion = build_model_and_criterion(cfg, device)
    print_run_header(cfg, args, model, device)

    if args.compile and device.type == "cuda":
        import platform
        if platform.system() == "Linux":
            print(f"Compiling model with mode='{args.compile_mode}' (first batch will be slow) …")
            model = torch.compile(model, mode=args.compile_mode)
        else:
            # Triton (required by inductor modes) has no Windows wheels;
            # aot_eager captures the graph and speeds up autograd without Triton.
            print("torch.compile: Triton not available on Windows — using backend='aot_eager'")
            model = torch.compile(model, backend="aot_eager")

    param_groups = [
        {"params": model.backbone.parameters(), "lr": tc["lr_backbone"]},
        {"params": [p for n, p in model.named_parameters() if "backbone" not in n], "lr": tc["lr"]},
    ]
    optimizer = AdamW(param_groups, weight_decay=tc["weight_decay"])
    warmup_ep = tc.get("warmup_epochs", 0)
    cosine_ep = max(tc["epochs"] - warmup_ep, 1)
    cosine = CosineAnnealingLR(optimizer, T_max=cosine_ep, eta_min=tc["lr"] * 1e-2)
    if warmup_ep > 0:
        warmup = LinearLR(optimizer, start_factor=0.01, end_factor=1.0, total_iters=warmup_ep)
        scheduler = SequentialLR(optimizer, schedulers=[warmup, cosine], milestones=[warmup_ep])
    else:
        scheduler = cosine
    # AMP: not supported on XLA (TPU uses bfloat16 natively without GradScaler)
    amp_enabled = tc["amp"] and device.type not in ("cpu", "xla")
    scaler = GradScaler(device.type, enabled=amp_enabled) if device.type != "xla" else None
    ema = EMA(model, tc["ema_decay"]) if tc["ema"] else None

    save_dir = Path(lc["save_dir"]) / lc["project"]
    save_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(save_dir / "tb")

    start_epoch = 0
    best_map = 0.0

    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        if "scheduler" in ckpt:
            scheduler.load_state_dict(ckpt["scheduler"])
        start_epoch = ckpt["epoch"] + 1
        best_map = ckpt.get("best_map", 0.0)
        print(f"Resumed from epoch {start_epoch}")

    n_train = len(train_loader.dataset)
    n_val   = len(val_loader.dataset) if val_loader else 0
    print(f"  Dataset    train {n_train:,}  │  val {n_val:,}  │  workers {num_workers}")
    print(f"  Save dir   {save_dir}")
    if args.resume:
        print(f"  Resuming   epoch {start_epoch} → {tc['epochs']}")
    print()
    print_epoch_header()

    for epoch in range(start_epoch, tc["epochs"]):
        if epoch != start_epoch and (epoch - start_epoch) % _HDREP == 0:
            print_epoch_header()

        current_lr = optimizer.param_groups[-1]["lr"]
        train_metrics = train_one_epoch(
            model, criterion, train_loader, optimizer, scaler, device, epoch,
            amp_enabled=amp_enabled,
            grad_accum=tc["grad_accumulate"],
            clip_norm=tc["clip_grad_norm"],
            mark_step_fn=mark_step_fn,
        )
        scheduler.step()

        if ema:
            ema.update(model)

        print_epoch_row(epoch, tc["epochs"], train_metrics, current_lr)

        for k, v in train_metrics.items():
            writer.add_scalar(f"train/{k}", v, epoch)

        if not args.no_val and (epoch + 1) % lc["val_period"] == 0:
            eval_model = ema.ema if ema else model
            val_metrics = validate(eval_model, val_loader, device, str(data_root / dc["val_ann"]), amp_enabled=amp_enabled)
            for k, v in val_metrics.items():
                writer.add_scalar(f"val/{k}", v, epoch)
            print_val_metrics(epoch, tc["epochs"], val_metrics)

            if val_metrics["mAP"] > best_map:
                best_map = val_metrics["mAP"]
                torch.save({"model": model.state_dict(), "epoch": epoch, "best_map": best_map}, save_dir / "best.pt")

        torch.save({
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "epoch": epoch,
            "best_map": best_map,
        }, save_dir / "last.pt")

        if (epoch + 1) % lc["save_period"] == 0:
            torch.save({
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "epoch": epoch,
                "best_map": best_map,
            }, save_dir / f"epoch_{epoch}.pt")

    print(_hline(_EW, "└", "┴", "┘"))
    writer.close()
    print(f"\n  Training complete — best mAP: {best_map:.4f}  │  checkpoints: {save_dir}")


if __name__ == "__main__":
    main()
