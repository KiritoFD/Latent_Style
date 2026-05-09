from __future__ import annotations

import argparse
import json
import logging
import os
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from .dataset import AdaCUTLatentDataset
from .trainer import SBTrainer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _set_cpu_threads(config: dict) -> None:
    train_cfg = config.get("training", {})
    cpu_threads = train_cfg.get("cpu_threads")
    cpu_interop_threads = train_cfg.get("cpu_interop_threads")
    if cpu_threads is not None:
        try:
            torch.set_num_threads(int(cpu_threads))
        except Exception:
            pass
    if cpu_interop_threads is not None:
        try:
            torch.set_num_interop_threads(int(cpu_interop_threads))
        except Exception:
            pass


def _seed_worker(worker_id: int) -> None:
    seed = torch.initial_seed() % (2**32)
    random.seed(seed)
    np.random.seed(seed)
    torch.set_num_threads(1)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train latent Schrodinger bridge model")
    parser.add_argument("--config", type=str, default="config.json", help="Path to config json")
    parser.add_argument("--resume", type=str, default=None, help="Optional checkpoint path to resume")
    args = parser.parse_args()

    config_path = Path(args.config).resolve()
    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    if args.resume:
        config.setdefault("training", {})
        config["training"]["resume_checkpoint"] = args.resume

    train_cfg = config.get("training", {})
    seed = int(train_cfg.get("seed", 42))
    _set_seed(seed)
    _set_cpu_threads(config)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)
    logger.info("Seed: %d", seed)

    data_cfg = config.get("data", {})
    dataset = AdaCUTLatentDataset(
        data_root=data_cfg.get("data_root", "../latent-256"),
        style_subdirs=data_cfg.get("style_subdirs", ["photo", "Hayao", "monet", "vangogh", "cezanne"]),
        allow_hflip=bool(data_cfg.get("allow_hflip", False)),
        identity_ratio=data_cfg.get("identity_ratio", None),
        batch_size_hint=int(train_cfg.get("batch_size", 24)),
        balance_target_styles_per_batch=bool(data_cfg.get("balance_target_styles_per_batch", True)),
        preload_to_gpu=bool(data_cfg.get("preload_to_gpu", False)),
        preload_max_vram_gb=float(data_cfg.get("preload_max_vram_gb", 0.0)),
        preload_reserve_ratio=float(data_cfg.get("preload_reserve_ratio", 0.35)),
        virtual_length_multiplier=int(data_cfg.get("virtual_length_multiplier", 1)),
        device=str(device),
    )

    style_count = len(dataset.style_subdirs)
    if int(config.get("model", {}).get("num_styles", style_count)) != style_count:
        logger.warning("model.num_styles mismatch detected; forcing to %d", style_count)
        config.setdefault("model", {})
        config["model"]["num_styles"] = style_count

    batch_size = int(train_cfg.get("batch_size", 24))
    num_workers = int(train_cfg.get("num_workers", 0))
    shuffle = bool(train_cfg.get("shuffle", False))
    persistent_workers = bool(train_cfg.get("persistent_workers", False) and num_workers > 0)
    pin_memory = bool(train_cfg.get("pin_memory", device.type == "cuda"))
    generator = torch.Generator().manual_seed(seed)

    dataloader_kwargs = dict(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        worker_init_fn=_seed_worker,
        generator=generator,
    )
    if num_workers > 0:
        dataloader_kwargs["persistent_workers"] = persistent_workers
        dataloader_kwargs["prefetch_factor"] = max(1, int(train_cfg.get("prefetch_factor", 2)))
    dataloader = DataLoader(**dataloader_kwargs)

    logger.info(
        "DataLoader | batch=%d workers=%d shuffle=%s pin_memory=%s persistent_workers=%s preload_to_gpu=%s balanced_target=%s",
        batch_size,
        num_workers,
        shuffle,
        pin_memory,
        persistent_workers,
        bool(getattr(dataset, "preload_to_gpu", False)),
        bool(getattr(dataset, "balance_target_styles_per_batch", False)),
    )

    trainer = SBTrainer(config=config, device=device, config_path=str(config_path))

    epoch = int(trainer.start_epoch)
    while epoch <= int(trainer.num_epochs):
        dataset.set_epoch(epoch)
        metrics = trainer.train_epoch(dataloader, epoch)
        trainer.step_scheduler()
        trainer.log_epoch(epoch, metrics)
        logger.info(
            "Epoch %d/%d | loss=%.4f flow=%.4f kin=%.4f ot=%.4f tswd=%.4f color=%.4f rep=%.4f ent=%.3f sigma=%.3f idr=%.2f t=%.3f |v|=%.3f lr=%.2e data=%.1fs comp=%.1fs",
            epoch,
            trainer.num_epochs,
            metrics.get("loss", 0.0),
            metrics.get("flow", 0.0),
            metrics.get("kinetic_energy", 0.0),
            metrics.get("ot_cost", 0.0),
            metrics.get("terminal_swd", 0.0),
            metrics.get("color", 0.0),
            metrics.get("repulsive", 0.0),
            metrics.get("plan_entropy", 0.0),
            metrics.get("bridge_sigma", 0.0),
            metrics.get("identity_ratio", 0.0),
            metrics.get("t_mean", 0.0),
            metrics.get("velocity_abs", 0.0),
            metrics.get("lr", 0.0),
            metrics.get("data_time_sec", 0.0),
            metrics.get("compute_time_sec", 0.0),
        )
        if epoch % trainer.save_interval == 0 or epoch == trainer.num_epochs:
            trainer.save_checkpoint(epoch, metrics)
        epoch += 1

    logger.info("Training completed.")


if __name__ == "__main__":
    main()
