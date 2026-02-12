from __future__ import annotations

import argparse
import json
import logging
import os
import random
from pathlib import Path
from typing import Iterable, List

import numpy as np
import torch
from torch.utils.data import DataLoader

try:
    from .dataset import AdaCUTLatentDataset
    from .trainer import AdaCUTTrainer
except ImportError:  # pragma: no cover
    from dataset import AdaCUTLatentDataset
    from trainer import AdaCUTTrainer

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
        except Exception:  # pragma: no cover
            pass
    else:
        # Default: Limit main process threads to avoid CPU saturation/overheating
        # especially in WSL or high-core environments.
        default_threads = 1
        torch.set_num_threads(default_threads)
        logger.info("Auto-set main process CPU threads to %d", default_threads)

    if cpu_interop_threads is not None:
        try:
            torch.set_num_interop_threads(int(cpu_interop_threads))
        except Exception:  # pragma: no cover
            pass


def _set_cpu_env_threads(config: dict) -> None:
    train_cfg = config.get("training", {})
    env_threads = train_cfg.get("cpu_env_threads")
    if env_threads is None:
        return
    try:
        threads = str(int(env_threads))
    except Exception:  # pragma: no cover
        return
    for key in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
    ):
        os.environ.setdefault(key, threads)


def _parse_cpu_affinity(value) -> List[int]:
    if value is None:
        return []
    if isinstance(value, int):
        return [value]
    if isinstance(value, (list, tuple)):
        return [int(v) for v in value]
    if isinstance(value, str):
        cpus: List[int] = []
        for part in value.split(","):
            part = part.strip()
            if not part:
                continue
            if "-" in part:
                start_s, end_s = part.split("-", 1)
                start, end = int(start_s), int(end_s)
                if start <= end:
                    cpus.extend(range(start, end + 1))
                else:
                    cpus.extend(range(end, start + 1))
            else:
                cpus.append(int(part))
        return cpus
    return []


def _apply_cpu_affinity(config: dict) -> None:
    if not hasattr(os, "sched_setaffinity"):
        return
    train_cfg = config.get("training", {})
    affinity = _parse_cpu_affinity(train_cfg.get("cpu_affinity"))
    if not affinity:
        return
    try:
        os.sched_setaffinity(0, set(affinity))
        logger.info("CPU affinity set to: %s", affinity)
    except Exception as exc:  # pragma: no cover
        logger.warning("Failed to set CPU affinity: %s", exc)


def _seed_worker(worker_id: int) -> None:
    seed = torch.initial_seed() % (2**32)
    random.seed(seed)
    np.random.seed(seed)
    # Workers should be single-threaded to prevent thread explosion (N workers * M threads)
    torch.set_num_threads(1)


def _resolve_num_workers(config: dict) -> int:
    train_cfg = config.get("training", {})
    preload = bool(config.get("data", {}).get("preload_to_gpu", False))
    if preload:
        return 0
    if train_cfg.get("num_workers") is not None:
        return int(train_cfg["num_workers"])
    # Optimization: For lightweight in-memory latent datasets, multiprocessing overhead (IPC)
    # often exceeds the cost of data fetching. Default to 0 to minimize CPU load/heat.
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Latent AdaCUT")
    parser.add_argument("--config", type=str, default="config.json", help="Path to config json")
    parser.add_argument("--resume", type=str, default=None, help="Optional checkpoint path to resume")
    args = parser.parse_args()

    config_path = Path(args.config).resolve()
    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    if args.resume:
        config.setdefault("training", {})
        config["training"]["resume_checkpoint"] = args.resume

    seed = int(config.get("training", {}).get("seed", 42))
    _set_seed(seed)
    _set_cpu_env_threads(config)
    _apply_cpu_affinity(config)
    _set_cpu_threads(config)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)
    logger.info("Seed: %d", seed)

    data_cfg = config.get("data", {})
    dataset = AdaCUTLatentDataset(
        data_root=data_cfg.get("data_root", "../../latents"),
        style_subdirs=data_cfg.get("style_subdirs", ["photo", "monet", "vangogh", "cezanne"]),
        content_style=data_cfg.get("content_style", "photo"),
        allow_hflip=bool(data_cfg.get("allow_hflip", True)),
        preload_to_gpu=bool(data_cfg.get("preload_to_gpu", False)),
        virtual_length_multiplier=int(data_cfg.get("virtual_length_multiplier", 4)),
        device=str(device),
    )
    style_count = len(dataset.style_subdirs)
    model_style_count = int(config.get("model", {}).get("num_styles", style_count))
    if model_style_count != style_count:
        logger.warning(
            "model.num_styles=%d does not match data.style_subdirs=%d, using %d",
            model_style_count,
            style_count,
            style_count,
        )
        config.setdefault("model", {})
        config["model"]["num_styles"] = style_count

    num_workers = _resolve_num_workers(config)
    preload_to_gpu = bool(data_cfg.get("preload_to_gpu", False))
    pin_memory_default = False
    pin_memory = bool(config.get("training", {}).get("pin_memory", pin_memory_default))
    persistent_workers = bool(config.get("training", {}).get("persistent_workers", True))
    batch_size = int(config.get("training", {}).get("batch_size", 64))
    
    # Optimization: If running in main process, pin_memory adds unnecessary CPU overhead
    if num_workers == 0:
        pin_memory = False

    dl_generator = torch.Generator()
    dl_generator.manual_seed(seed)

    dataloader_kwargs = dict(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=num_workers,
        pin_memory=pin_memory and (not preload_to_gpu),
        worker_init_fn=_seed_worker,
        generator=dl_generator,
    )
    if num_workers > 0:
        dataloader_kwargs["persistent_workers"] = persistent_workers
        dataloader_kwargs["prefetch_factor"] = max(1, int(config.get("training", {}).get("prefetch_factor", 2)))
    dataloader = DataLoader(**dataloader_kwargs)
    logger.info(
        "DataLoader | batch=%d workers=%d pin_memory=%s persistent_workers=%s preload_to_gpu=%s",
        batch_size,
        num_workers,
        pin_memory and (not preload_to_gpu),
        persistent_workers if num_workers > 0 else False,
        preload_to_gpu,
    )

    trainer = AdaCUTTrainer(config=config, device=device, config_path=str(config_path))

    for epoch in range(trainer.start_epoch, trainer.num_epochs + 1):
        dataset.set_epoch(epoch)
        metrics = trainer.train_epoch(dataloader, epoch)
        trainer.step_scheduler()
        trainer.log_epoch(epoch, metrics)

        logger.info(
            "Epoch %d/%d | loss=%.4f code=%.4f cpn=%.3f crn=%.3f cycle=%.4f gram=%.4f sgram=%.4f gramw=%.4f moment=%.4f cmoment=%.4f push=%.4f dtv=%.4f stv=%.4f nce=%.4f semi=%.4f idt=%.4f wcyc=%.2f wnce=%.2f wsemi=%.2f widt=%.2f xfer=%.2f lr=%.2e",
            epoch,
            trainer.num_epochs,
            metrics["loss"],
            metrics.get("code", 0.0),
            metrics.get("code_pred_norm", 0.0),
            metrics.get("code_ref_norm", 0.0),
            metrics.get("cycle", 0.0),
            metrics.get("gram", 0.0),
            metrics.get("stroke_gram", 0.0),
            metrics.get("gram_w", 0.0),
            metrics.get("moment", 0.0),
            metrics.get("color_moment", 0.0),
            metrics.get("push", 0.0),
            metrics.get("delta_tv", 0.0),
            metrics.get("style_spatial_tv", 0.0),
            metrics.get("nce", 0.0),
            metrics.get("semigroup", 0.0),
            metrics.get("idt", 0.0),
            metrics.get("w_cycle_eff", 0.0),
            metrics.get("w_nce_eff", 0.0),
            metrics.get("w_semigroup_eff", 0.0),
            metrics.get("w_idt_eff", 0.0),
            metrics.get("transfer_ratio", 0.0),
            metrics["lr"],
        )

        ckpt_path = None
        if epoch % trainer.save_interval == 0 or epoch == trainer.num_epochs:
            ckpt_path = trainer.save_checkpoint(epoch, metrics)

        do_full_eval = False
        if trainer.full_eval_interval > 0 and (epoch % trainer.full_eval_interval == 0):
            do_full_eval = True
        if trainer.run_full_eval_on_last_epoch and (epoch == trainer.num_epochs):
            do_full_eval = True
        if do_full_eval:
            if ckpt_path is None:
                ckpt_path = trainer.save_checkpoint(epoch, metrics)
            trainer.run_full_evaluation(epoch, checkpoint_path=ckpt_path)

    logger.info("Training completed.")


if __name__ == "__main__":
    main()
