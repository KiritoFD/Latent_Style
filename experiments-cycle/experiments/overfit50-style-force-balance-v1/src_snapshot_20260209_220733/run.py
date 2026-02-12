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
    if cpu_interop_threads is not None:
        try:
            torch.set_num_interop_threads(int(cpu_interop_threads))
        except Exception:  # pragma: no cover
            pass


def _seed_worker(worker_id: int) -> None:
    seed = torch.initial_seed() % (2**32)
    random.seed(seed)
    np.random.seed(seed)


def _resolve_num_workers(config: dict) -> int:
    train_cfg = config.get("training", {})
    preload = bool(config.get("data", {}).get("preload_to_gpu", False))
    if preload:
        return 0
    if train_cfg.get("num_workers") is not None:
        return int(train_cfg["num_workers"])
    cpu_count = os.cpu_count() or 1
    return max(2, cpu_count // 2)


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
    pin_memory_default = bool(torch.cuda.is_available() and (not preload_to_gpu))
    pin_memory = bool(config.get("training", {}).get("pin_memory", pin_memory_default))
    persistent_workers = bool(config.get("training", {}).get("persistent_workers", True))
    batch_size = int(config.get("training", {}).get("batch_size", 64))

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
            "Epoch %d/%d | loss=%.4f code=%.4f cpn=%.3f crn=%.3f cycle=%.4f gram=%.4f gramw=%.4f moment=%.4f push=%.4f nce=%.4f idt=%.4f wcyc=%.2f wnce=%.2f widt=%.2f xfer=%.2f lr=%.2e",
            epoch,
            trainer.num_epochs,
            metrics["loss"],
            metrics.get("code", 0.0),
            metrics.get("code_pred_norm", 0.0),
            metrics.get("code_ref_norm", 0.0),
            metrics.get("cycle", 0.0),
            metrics.get("gram", 0.0),
            metrics.get("gram_w", 0.0),
            metrics.get("moment", 0.0),
            metrics.get("push", 0.0),
            metrics.get("nce", 0.0),
            metrics.get("idt", 0.0),
            metrics.get("w_cycle_eff", 0.0),
            metrics.get("w_nce_eff", 0.0),
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
