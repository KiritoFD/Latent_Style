from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset

from dataset import AdaCUTLatentDataset
from trainer import AdaCUTTrainer


def _round_bs(x: float) -> int:
    bs = max(8, int(round(x / 8.0) * 8))
    return bs


def run_one(config_path: Path, target_gb: float = 10.5) -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    config["training"]["use_tqdm"] = False
    config["training"]["log_interval"] = 0
    config["training"]["num_epochs"] = 1
    config["training"]["full_eval_interval"] = 0
    config["training"]["full_eval_on_last_epoch"] = False

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_cfg = config.get("data", {})
    dataset = AdaCUTLatentDataset(
        data_root=data_cfg.get("data_root", "../../latents"),
        style_subdirs=data_cfg.get("style_subdirs", ["photo", "monet", "vangogh", "cezanne"]),
        clip_feature_root=data_cfg.get("clip_feature_root", "../../clip-feats-vitb32"),
        allow_hflip=bool(data_cfg.get("allow_hflip", True)),
        preload_to_gpu=bool(data_cfg.get("preload_to_gpu", False)),
        preload_max_vram_gb=float(data_cfg.get("preload_max_vram_gb", 0.0)),
        preload_reserve_ratio=float(data_cfg.get("preload_reserve_ratio", 0.35)),
        virtual_length_multiplier=int(data_cfg.get("virtual_length_multiplier", 4)),
        device=str(device),
    )
    dataset.set_epoch(1)

    half_len = max(1, len(dataset) // 2)
    subset = Subset(dataset, list(range(half_len)))
    bs = int(config["training"].get("batch_size", 64))
    dataloader = DataLoader(
        subset,
        batch_size=bs,
        shuffle=False,
        drop_last=True,
        num_workers=0,
        pin_memory=bool(config.get("training", {}).get("pin_memory", device.type == "cuda")),
    )

    trainer = AdaCUTTrainer(config=config, device=device, config_path=str(config_path))
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    trainer.train_epoch(dataloader, epoch=1)

    if device.type == "cuda":
        peak_alloc_gb = float(torch.cuda.max_memory_allocated() / (1024**3))
        peak_reserved_gb = float(torch.cuda.max_memory_reserved() / (1024**3))
    else:
        peak_alloc_gb = 0.0
        peak_reserved_gb = 0.0

    suggest = _round_bs(bs * target_gb / max(peak_reserved_gb, 1e-6))
    return {
        "config": str(config_path),
        "batch_size": bs,
        "peak_alloc_gb": peak_alloc_gb,
        "peak_reserved_gb": peak_reserved_gb,
        "suggest_batch_size_for_10p5g": suggest,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Half-epoch VRAM calibrator")
    parser.add_argument("--glob", type=str, default="config_S*.json")
    parser.add_argument("--target_gb", type=float, default=10.5)
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()

    root = Path(__file__).resolve().parent
    targets = sorted(root.glob(args.glob))
    if not targets:
        raise FileNotFoundError(f"No config matched: {args.glob}")

    rows = []
    for p in targets:
        row = run_one(p, target_gb=float(args.target_gb))
        rows.append(row)
        if args.apply:
            with open(p, "r", encoding="utf-8") as f:
                cfg = json.load(f)
            old_bs = int(cfg["training"].get("batch_size", row["batch_size"]))
            new_bs = int(row["suggest_batch_size_for_10p5g"])
            if new_bs != old_bs:
                lr = float(cfg["training"].get("learning_rate", 0.0))
                min_lr = float(cfg["training"].get("min_learning_rate", 0.0))
                scale = math.sqrt(new_bs / max(old_bs, 1))
                cfg["training"]["batch_size"] = new_bs
                if lr > 0:
                    cfg["training"]["learning_rate"] = lr * scale
                if min_lr > 0:
                    cfg["training"]["min_learning_rate"] = min_lr * scale
                with open(p, "w", encoding="utf-8") as f:
                    json.dump(cfg, f, indent=4, ensure_ascii=False)
                row["applied_new_batch_size"] = new_bs
                row["applied_lr_scale"] = scale
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
