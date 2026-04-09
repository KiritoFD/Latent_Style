from __future__ import annotations

import argparse
import inspect
import json
import logging
import os
import random
from pathlib import Path
from typing import List

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

_ALLOWED_LOSS_KEYS = {
    "w_color",
    "w_oob",
    "oob_threshold",
    "w_repulsive",
    "repulsive_margin",
    "repulsive_temperature",
    "repulsive_mode",
    "w_swd",
    "w_swd_unified",
    "w_swd_micro",
    "w_swd_macro",
    "swd_use_high_freq",
    "swd_hf_weight_ratio",
    "w_identity",
    "idr",
    "swd_patch_sizes",
    "swd_num_projections",
    "swd_projection_chunk_size",
    "swd_distance_mode",
    "swd_cdf_num_bins",
    "swd_cdf_tau",
    "swd_cdf_sample_size",
    "swd_cdf_bin_chunk_size",
    "swd_cdf_sample_chunk_size",
    "swd_batch_size",
}
_FORBIDDEN_LOSS_KEYS = {"w_distill", "distill_low_only", "distill_cross_domain_only", "w_code", "style_loss_source"}
_LOSS_WEIGHT_KEYS = (
    "w_swd",
    "w_swd_unified",
    "w_swd_micro",
    "w_swd_macro",
    "w_repulsive",
    "w_color",
    "w_oob",
    "w_identity",
)


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _configure_cuda_allocator(config: dict) -> None:
    if not torch.cuda.is_available():
        return
    train_cfg = config.get("training", {})
    alloc_conf = str(train_cfg.get("cuda_alloc_conf", "")).strip()
    if not alloc_conf:
        # Stable fallback for platforms where expandable_segments is unsupported.
        alloc_conf = "max_split_size_mb:128,garbage_collection_threshold:0.8"
    current = os.environ.get("PYTORCH_ALLOC_CONF", "").strip()
    if current:
        logger.info("Use existing PYTORCH_ALLOC_CONF=%s", current)
        return
    legacy = os.environ.get("PYTORCH_CUDA_ALLOC_CONF", "").strip()
    if legacy:
        os.environ["PYTORCH_ALLOC_CONF"] = legacy
        logger.info("Migrated PYTORCH_CUDA_ALLOC_CONF -> PYTORCH_ALLOC_CONF=%s", legacy)
        return
    os.environ["PYTORCH_ALLOC_CONF"] = alloc_conf
    logger.info("Set PYTORCH_ALLOC_CONF=%s", alloc_conf)


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
        # Keep PyTorch default thread policy unless explicitly configured.
        pass

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


def _resolve_num_workers(config: dict, *, preload_to_gpu: bool = False, dataset_bytes: int = 0) -> int:
    train_cfg = config.get("training", {})
    if preload_to_gpu:
        return 0

    def _resolve_windows_mp_mode() -> str:
        raw = train_cfg.get("windows_allow_multiprocess_dataloader", "auto")
        if isinstance(raw, bool):
            return "on" if raw else "off"
        mode = str(raw).strip().lower()
        if mode in {"on", "off", "auto"}:
            return mode
        return "auto"

    def _resolve_windows_auto_workers(requested_workers: int) -> int:
        mode = _resolve_windows_mp_mode()
        if mode == "off":
            return 0
        if mode == "on":
            return max(0, int(requested_workers))

        # auto: conservative enable only when dataset footprint is small enough.
        ds_gb = float(max(0, int(dataset_bytes))) / float(1024**3)
        limit_gb = float(train_cfg.get("windows_auto_mp_dataset_limit_gb", 0.75))
        max_workers = max(1, int(train_cfg.get("windows_auto_mp_max_workers", 2)))
        if ds_gb <= limit_gb:
            chosen = max(1, min(int(requested_workers), max_workers))
            logger.info(
                "Windows DataLoader auto mode: use num_workers=%d (requested=%d, dataset=%.2fGB, limit=%.2fGB).",
                chosen,
                requested_workers,
                ds_gb,
                limit_gb,
            )
            return chosen
        logger.warning(
            "Windows DataLoader auto mode: force num_workers=0 for stability "
            "(requested=%d, dataset=%.2fGB > limit %.2fGB).",
            requested_workers,
            ds_gb,
            limit_gb,
        )
        return 0

    requested = train_cfg.get("num_workers")
    if requested is not None:
        requested = int(requested)
        if requested >= 0:
            if os.name == "nt" and requested > 0:
                return _resolve_windows_auto_workers(requested)
            return requested
    cpu_count = os.cpu_count() or 4
    # -1/None means auto.
    if os.name == "nt":
        return _resolve_windows_auto_workers(2)
    return max(2, min(8, cpu_count // 2))

def _validate_loss_config(config: dict) -> None:
    loss_cfg = config.get("loss", {})
    if not isinstance(loss_cfg, dict):
        raise ValueError("config.loss must be a JSON object")
    forbidden = sorted(k for k in loss_cfg.keys() if k in _FORBIDDEN_LOSS_KEYS)
    if forbidden:
        raise ValueError(f"Removed loss key(s) found in config.loss: {forbidden}")
    unknown = sorted(k for k in loss_cfg.keys() if k not in _ALLOWED_LOSS_KEYS)
    if unknown:
        raise ValueError(f"Unknown loss key(s) in config.loss: {unknown}")

def _log_active_losses(config: dict) -> None:
    loss_cfg = config.get("loss", {})
    active = [k for k in _LOSS_WEIGHT_KEYS if float(loss_cfg.get(k, 0.0)) > 0.0]
    logger.info("Active losses: %s", ", ".join(active) if active else "(none)")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Latent AdaCUT")
    parser.add_argument("--config", type=str, default="config.json", help="Path to config json")
    parser.add_argument("--resume", type=str, default=None, help="Optional checkpoint path to resume")
    args = parser.parse_args()

    config_path = Path(args.config).resolve()
    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)
    _validate_loss_config(config)
    _log_active_losses(config)

    if args.resume:
        config.setdefault("training", {})
        config["training"]["resume_checkpoint"] = args.resume

    _configure_cuda_allocator(config)
    seed = int(config.get("training", {}).get("seed", 42))
    if bool(config.get("training", {}).get("cuda_sync_debug", False)):
        logger.warning("training.cuda_sync_debug=True will heavily reduce throughput and can hide batch-size scaling effects.")
    _set_seed(seed)
    _set_cpu_env_threads(config)
    _apply_cpu_affinity(config)
    _set_cpu_threads(config)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)
    logger.info("Seed: %d", seed)

    data_cfg = config.get("data", {})
    loss_cfg = config.get("loss", {})
    if "idr" in loss_cfg and "identity_ratio" not in data_cfg:
        idr_val = float(loss_cfg.get("idr", 0.0))
        # idr <= 0 means "no forced override": keep dataset's natural sampling ratio.
        if idr_val > 0.0:
            config.setdefault("data", {})
            config["data"]["identity_ratio"] = idr_val
            data_cfg = config.get("data", {})
    dataset_kwargs = dict(
        data_root=data_cfg.get("data_root", "../../latents"),
        style_subdirs=data_cfg.get("style_subdirs", ["photo", "monet", "vangogh", "cezanne"]),
        allow_hflip=bool(data_cfg.get("allow_hflip", True)),
        preload_to_gpu=bool(data_cfg.get("preload_to_gpu", False)),
        preload_max_vram_gb=float(data_cfg.get("preload_max_vram_gb", 0.0)),
        preload_reserve_ratio=float(data_cfg.get("preload_reserve_ratio", 0.35)),
        virtual_length_multiplier=int(data_cfg.get("virtual_length_multiplier", 4)),
        device=str(device),
    )
    identity_ratio = data_cfg.get("identity_ratio", None)
    if "identity_ratio" in inspect.signature(AdaCUTLatentDataset.__init__).parameters:
        dataset_kwargs["identity_ratio"] = identity_ratio
    elif identity_ratio is not None:
        logger.warning(
            "Dataset class does not support identity_ratio; falling back to dataset natural sampling. "
            "Please sync dataset.py and run.py to the same revision."
        )
    dataset = AdaCUTLatentDataset(**dataset_kwargs)
    dataset_bytes = 0
    try:
        dataset_bytes = int(dataset._estimate_dataset_bytes())  # type: ignore[attr-defined]
    except Exception:
        dataset_bytes = 0
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

    preload_to_gpu = bool(getattr(dataset, "preload_to_gpu", False))
    num_workers = _resolve_num_workers(config, preload_to_gpu=preload_to_gpu, dataset_bytes=dataset_bytes)
    pin_memory_default = (device.type == "cuda") and (not preload_to_gpu)
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

    epoch = int(trainer.start_epoch)
    while epoch <= int(trainer.num_epochs):
        dataset.set_epoch(epoch)
        metrics = trainer.train_epoch(dataloader, epoch)
        trainer.step_scheduler()
        trainer.log_epoch(epoch, metrics)

        logger.info(
            "Epoch %d/%d | loss=%.4f swd=%.4f rep=%.4f color=%.4f idt=%.4f idr=%.2f aent=%.3f amax=%.3f lr=%.2e data=%.1fs comp=%.1fs",
            epoch,
            trainer.num_epochs,
            metrics["loss"],
            metrics.get("swd", 0.0),
            metrics.get("repulsive", 0.0),
            metrics.get("color", 0.0),
            metrics.get("identity", 0.0),
            metrics.get("identity_ratio", 0.0),
            metrics.get("aent", 0.0),
            metrics.get("amax", 0.0),
            metrics["lr"],
            metrics.get("data_time_sec", 0.0),
            metrics.get("compute_time_sec", 0.0),
        )

        if epoch % trainer.save_interval == 0 or epoch == trainer.num_epochs:
            trainer.save_checkpoint(epoch, metrics)
        epoch += 1

    logger.info("Training completed.")


if __name__ == "__main__":
    main()
