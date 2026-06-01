from __future__ import annotations

import argparse
import logging
import os
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from config_schema import ExperimentConfig, load_experiment_config
from trainer import SBTrainer
from utils.dataset import AdaCUTLatentDataset

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


def _set_cpu_threads(config: ExperimentConfig) -> None:
    train_cfg = config.training
    cpu_threads = train_cfg.cpu_threads
    cpu_interop_threads = train_cfg.cpu_interop_threads
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


def _resolve_num_workers(requested: int) -> int:
    if requested >= 0:
        return requested
    if os.name == "nt":
        return 0
    cpu_count = os.cpu_count() or 4
    return max(2, min(8, cpu_count // 2))


def main() -> None:
    parser = argparse.ArgumentParser(description="Train latent Schrodinger bridge model")
    parser.add_argument("--config", type=str, default="config.json", help="Path to config json")
    parser.add_argument("--resume", type=str, default=None, help="Optional checkpoint path to resume")
    args = parser.parse_args()

    config_path = Path(args.config).resolve()
    config = load_experiment_config(config_path)

    if args.resume:
        config.training.resume_checkpoint = args.resume

    train_cfg = config.training
    seed = int(train_cfg.seed)
    _set_seed(seed)
    _set_cpu_threads(config)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)
    logger.info("Seed: %d", seed)

    data_cfg = config.data
    dataset = AdaCUTLatentDataset(
        data_root=data_cfg.data_root,
        style_subdirs=data_cfg.style_subdirs,
        allow_hflip=bool(data_cfg.allow_hflip),
        identity_ratio=data_cfg.identity_ratio,
        batch_size_hint=int(train_cfg.batch_size),
        balance_target_styles_per_batch=bool(data_cfg.balance_target_styles_per_batch),
        preload_to_gpu=bool(data_cfg.preload_to_gpu),
        preload_max_vram_gb=float(data_cfg.preload_max_vram_gb),
        preload_reserve_ratio=float(data_cfg.preload_reserve_ratio),
        virtual_length_multiplier=float(data_cfg.virtual_length_multiplier),
        content_style_sampling_weights=data_cfg.content_style_sampling_weights,
        target_style_sampling_weights=data_cfg.target_style_sampling_weights,
        pairing_cache_path=data_cfg.pairing_cache_path,
        pairing_cache_topk=int(data_cfg.pairing_cache_topk),
        pairing_cache_sample_mode=str(data_cfg.pairing_cache_sample_mode),
        pairing_cache_cross_only=bool(data_cfg.pairing_cache_cross_only),
        device=str(device),
    )

    style_count = len(dataset.style_subdirs)
    if int(config.model.num_styles) != style_count:
        logger.warning("model.num_styles mismatch detected; forcing to %d", style_count)
        config.model.num_styles = style_count

    batch_size = int(train_cfg.batch_size)
    num_workers = _resolve_num_workers(int(train_cfg.num_workers))
    shuffle = bool(train_cfg.shuffle)
    persistent_workers = bool(train_cfg.persistent_workers and num_workers > 0)
    pin_memory = bool(train_cfg.pin_memory)
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
        dataloader_kwargs["prefetch_factor"] = max(1, int(train_cfg.prefetch_factor))
    dataloader = DataLoader(**dataloader_kwargs)

    logger.info(
        "DataLoader | batch=%d workers=%d shuffle=%s pin_memory=%s persistent_workers=%s preload_to_gpu=%s balanced_target=%s pairing_cache=%s pairing_mode=%s topk=%d",
        batch_size,
        num_workers,
        shuffle,
        pin_memory,
        persistent_workers,
        bool(getattr(dataset, "preload_to_gpu", False)),
        bool(getattr(dataset, "balance_target_styles_per_batch", False)),
        bool(getattr(dataset, "offline_pairing_map", {})),
        str(getattr(dataset, "pairing_cache_sample_mode", "")),
        int(getattr(dataset, "pairing_cache_topk", 0)),
    )

    trainer = SBTrainer(config=config, device=device, config_path=str(config_path))

    epoch = int(trainer.start_epoch)
    while epoch <= int(trainer.num_epochs):
        dataset.set_epoch(epoch)
        metrics = trainer.train_epoch(dataloader, epoch)
        trainer.step_scheduler()
        trainer.log_epoch(epoch, metrics)
        logger.info(
            "Epoch %d/%d | loss=%.4f flow=%.4f kin=%.4f ot=%.4f tswd=%.4f attn=%.3f k=%.3f ent=%.3f sigma=%.3f idr=%.2f t=%.3f |v|=%.3f lr=%.2e data=%.1fs comp=%.1fs",
            epoch,
            trainer.num_epochs,
            metrics.get("loss", 0.0),
            metrics.get("flow", 0.0),
            metrics.get("kinetic_energy", 0.0),
            metrics.get("ot_cost", 0.0),
            metrics.get("terminal_swd", 0.0),
            metrics.get("semantic_attn_mean", 0.0),
            metrics.get("semantic_k_abs", 0.0),
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
