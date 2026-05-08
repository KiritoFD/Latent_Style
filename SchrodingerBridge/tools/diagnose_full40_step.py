from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    repo = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo / "src"))

    from dataset import AdaCUTLatentDataset
    from trainer import SBTrainer

    config_path = Path(args.config).resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}")

    data_root_raw = Path(config["data"]["data_root"]).expanduser()
    if data_root_raw.is_absolute():
        data_root = data_root_raw
    else:
        data_root = (repo / data_root_raw).resolve()

    dataset = AdaCUTLatentDataset(
        data_root=str(data_root),
        style_subdirs=config["data"]["style_subdirs"],
        allow_hflip=bool(config["data"].get("allow_hflip", False)),
        identity_ratio=config["data"].get("identity_ratio", None),
        batch_size_hint=int(config["training"]["batch_size"]),
        balance_target_styles_per_batch=bool(config["data"].get("balance_target_styles_per_batch", True)),
        preload_to_gpu=False,
        preload_max_vram_gb=float(config["data"].get("preload_max_vram_gb", 0.0)),
        preload_reserve_ratio=float(config["data"].get("preload_reserve_ratio", 0.35)),
        virtual_length_multiplier=int(config["data"].get("virtual_length_multiplier", 1)),
        device=str(device),
    )
    print(f"dataset_len={len(dataset)}")

    loader = DataLoader(
        dataset,
        batch_size=int(config["training"]["batch_size"]),
        shuffle=False,
        drop_last=True,
        num_workers=0,
        pin_memory=True,
    )

    trainer = SBTrainer(config=config, device=device, config_path=str(config_path))

    t0 = time.time()
    batch = next(iter(loader))
    print(f"load_batch_sec={time.time() - t0:.3f}")

    content = batch["content"].to(device, non_blocking=True)
    style = batch["target_style"].to(device, non_blocking=True)
    target_style_id = batch["target_style_id"].to(device, non_blocking=True)
    source_style_id = batch.get("source_style_id")
    if source_style_id is not None:
        source_style_id = source_style_id.to(device, non_blocking=True)

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)

    trainer.optimizer.zero_grad(set_to_none=True)

    t1 = time.time()
    if device.type == "cuda":
        autocast_ctx = torch.amp.autocast("cuda", enabled=trainer.use_amp, dtype=trainer.amp_dtype)
    else:
        autocast_ctx = torch.autocast("cpu", enabled=False)
    with autocast_ctx:
        losses = trainer.loss_fn.compute(
            trainer.model,
            content=content,
            target_style=style,
            target_style_id=target_style_id,
            source_style_id=source_style_id,
        )
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    print(f"forward_sec={time.time() - t1:.3f}")
    print(f"loss={float(losses['loss']):.6f}")
    if device.type == "cuda":
        print(f"forward_peak_mem_mb={torch.cuda.max_memory_allocated(device) / 1024 / 1024:.1f}")

    t2 = time.time()
    trainer.scaler.scale(losses["loss"]).backward()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    print(f"backward_sec={time.time() - t2:.3f}")
    if device.type == "cuda":
        print(f"backward_peak_mem_mb={torch.cuda.max_memory_allocated(device) / 1024 / 1024:.1f}")


if __name__ == "__main__":
    main()
