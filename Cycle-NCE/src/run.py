from __future__ import annotations

import argparse
import json
import logging
import os
import random
import math
import atexit
import signal
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


class _SingleRunLock:
    def __init__(self, lock_path: Path) -> None:
        self.lock_path = Path(lock_path)
        self._fd: int | None = None

    def acquire(self) -> None:
        self.lock_path.parent.mkdir(parents=True, exist_ok=True)
        self._remove_stale_lock_if_needed()
        try:
            self._fd = os.open(str(self.lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.write(self._fd, str(os.getpid()).encode("utf-8"))
            os.fsync(self._fd)
        except FileExistsError as exc:
            pid_text = ""
            try:
                pid_text = self.lock_path.read_text(encoding="utf-8").strip()
            except Exception:
                pid_text = "unknown"
            raise RuntimeError(
                f"Another training process is running (lock: {self.lock_path}, pid: {pid_text}). "
                "Clean stale lock only if that process is gone."
            ) from exc

        def _cleanup(*_args):
            self.release()

        atexit.register(_cleanup)
        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                signal.signal(sig, _cleanup)
            except Exception:
                pass

    def release(self) -> None:
        if self._fd is not None:
            try:
                os.close(self._fd)
            except Exception:
                pass
            self._fd = None
        try:
            if self.lock_path.exists():
                self.lock_path.unlink()
        except Exception:
            pass

    @staticmethod
    def _pid_alive(pid: int) -> bool:
        if pid <= 0:
            return False
        try:
            os.kill(pid, 0)
            return True
        except ProcessLookupError:
            return False
        except PermissionError:
            # Process exists but we don't own it.
            return True
        except Exception:
            return False

    def _remove_stale_lock_if_needed(self) -> None:
        if not self.lock_path.exists():
            return
        try:
            pid_text = self.lock_path.read_text(encoding="utf-8").strip()
            pid = int(pid_text)
        except Exception:
            pid = -1
        if not self._pid_alive(pid):
            try:
                self.lock_path.unlink()
                logger.warning("Removed stale run lock: %s (pid=%s)", self.lock_path, pid if pid > 0 else "unknown")
            except Exception:
                pass


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _seed_worker(worker_id: int) -> None:
    seed = torch.initial_seed() % (2**32)
    random.seed(seed)
    np.random.seed(seed)


def _resolve_worker_cpu_pool(train_cfg: dict) -> list[int]:
    """
    Build a CPU-id pool for DataLoader workers.
    On Linux/WSL, this respects current process affinity.
    """
    if hasattr(os, "sched_getaffinity"):
        cpu_ids = sorted(int(x) for x in os.sched_getaffinity(0))
    else:
        cpu_count = os.cpu_count() or 1
        cpu_ids = list(range(cpu_count))

    budget = int(train_cfg.get("dataloader_cpu_budget", 0))
    if budget <= 0 or budget >= len(cpu_ids):
        return cpu_ids

    # Evenly sample cores from allowed set to spread thermal load.
    out: list[int] = []
    used = set()
    n = len(cpu_ids)
    for i in range(budget):
        idx = int(math.floor(i * n / budget))
        idx = max(0, min(n - 1, idx))
        cid = cpu_ids[idx]
        if cid not in used:
            out.append(cid)
            used.add(cid)
    for cid in cpu_ids:
        if len(out) >= budget:
            break
        if cid not in used:
            out.append(cid)
            used.add(cid)
    return out


def _make_worker_init_fn(seed: int, train_cfg: dict, num_workers: int, cpu_pool: list[int]):
    affinity_mode = str(train_cfg.get("dataloader_affinity_mode", "none")).lower()
    single_thread = bool(train_cfg.get("dataloader_worker_single_thread", True))
    threads_per_worker = max(1, int(train_cfg.get("dataloader_threads_per_worker", 1)))

    def _worker_init(worker_id: int) -> None:
        base = (int(seed) + int(worker_id)) % (2**32)
        random.seed(base)
        np.random.seed(base)

        if single_thread:
            os.environ["OMP_NUM_THREADS"] = str(threads_per_worker)
            os.environ["MKL_NUM_THREADS"] = str(threads_per_worker)
            torch.set_num_threads(threads_per_worker)
            try:
                torch.set_num_interop_threads(1)
            except RuntimeError:
                pass

        if affinity_mode != "none" and cpu_pool and hasattr(os, "sched_setaffinity"):
            try:
                if affinity_mode == "compact":
                    cpu_id = cpu_pool[worker_id % len(cpu_pool)]
                else:
                    # spread: stride through pool to reduce hotspot concentration
                    stride = max(1, len(cpu_pool) // max(1, num_workers))
                    cpu_id = cpu_pool[(worker_id * stride) % len(cpu_pool)]
                os.sched_setaffinity(0, {int(cpu_id)})
            except Exception:
                pass

    return _worker_init


def _resolve_num_workers(config: dict) -> int:
    train_cfg = config.get("training", {})
    preload = bool(config.get("data", {}).get("preload_to_gpu", False))
    if preload:
        return 0
    if train_cfg.get("num_workers") is not None:
        num_workers = int(train_cfg["num_workers"])
    else:
        cpu_count = os.cpu_count() or 1
        num_workers = max(2, cpu_count // 2)
    budget = int(train_cfg.get("dataloader_cpu_budget", 0))
    if budget > 0:
        num_workers = min(num_workers, budget)
    num_workers = max(0, num_workers)
    return num_workers


def _maybe_auto_preload_latents(dataset: AdaCUTLatentDataset, config: dict, device: torch.device) -> bool:
    train_cfg = config.get("training", {})
    data_cfg = config.get("data", {})
    if bool(data_cfg.get("preload_to_gpu", False)):
        return bool(dataset.preload_to_gpu)
    if device.type != "cuda":
        return bool(dataset.preload_to_gpu)
    if not bool(train_cfg.get("auto_preload_latents_to_gpu", False)):
        return bool(dataset.preload_to_gpu)
    budget_mb = float(train_cfg.get("auto_preload_gpu_budget_mb", 2048))
    need_mb = float(getattr(dataset, "total_bytes", 0)) / (1024.0 * 1024.0)
    if need_mb <= max(0.0, budget_mb):
        dataset.preload_to_device(str(device))
        return True
    logger.info(
        "Auto preload skipped: need %.1fMB > budget %.1fMB",
        need_mb,
        budget_mb,
    )
    return bool(dataset.preload_to_gpu)


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

    lock_path = config.get("training", {}).get("run_lock_path", "../run.lock")
    lock = _SingleRunLock((config_path.parent / str(lock_path)).resolve())
    lock.acquire()

    seed = int(config.get("training", {}).get("seed", 42))
    _set_seed(seed)

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
    auto_preloaded = _maybe_auto_preload_latents(dataset, config, device)
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

    if auto_preloaded:
        config.setdefault("data", {})
        config["data"]["preload_to_gpu"] = True
    num_workers = _resolve_num_workers(config)
    train_cfg = config.get("training", {})
    cpu_pool = _resolve_worker_cpu_pool(train_cfg)
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
        worker_init_fn=_make_worker_init_fn(seed=seed, train_cfg=train_cfg, num_workers=num_workers, cpu_pool=cpu_pool),
        generator=dl_generator,
    )
    if num_workers > 0:
        dataloader_kwargs["persistent_workers"] = persistent_workers
        dataloader_kwargs["prefetch_factor"] = max(1, int(config.get("training", {}).get("prefetch_factor", 2)))
    dataloader = DataLoader(**dataloader_kwargs)
    logger.info(
        "DataLoader | batch=%d workers=%d pin_memory=%s persistent_workers=%s preload_to_gpu=%s affinity=%s cpu_pool=%d worker_threads=%d",
        batch_size,
        num_workers,
        pin_memory and (not preload_to_gpu),
        persistent_workers if num_workers > 0 else False,
        preload_to_gpu,
        str(train_cfg.get("dataloader_affinity_mode", "none")).lower(),
        len(cpu_pool),
        max(1, int(train_cfg.get("dataloader_threads_per_worker", 1))),
    )

    trainer = AdaCUTTrainer(config=config, device=device, config_path=str(config_path))

    for epoch in range(trainer.start_epoch, trainer.num_epochs + 1):
        dataset.set_epoch(epoch)
        metrics = trainer.train_epoch(dataloader, epoch)
        trainer.step_scheduler()
        trainer.log_epoch(epoch, metrics)

        logger.info(
            "Epoch %d/%d | loss=%.4f cls=%.4f p=%.4f pm=%.4f pw=%.3f p_t=%.3f hard=%.3f margin=%.4f proto_cos=%.4f feat=%.4f cycle=%.4f idt=%.4f data_wait=%.1fms step_wall=%.1fms lr=%.2e",
            epoch,
            trainer.num_epochs,
            metrics["loss"],
            metrics.get("style_ce", 0.0),
            metrics.get("prob", 0.0),
            metrics.get("prob_margin", 0.0),
            metrics.get("prob_weight_mean", 0.0),
            metrics.get("cls_target_prob", 0.0),
            metrics.get("cls_hard_ratio", 0.0),
            metrics.get("xfer_margin", 0.0),
            metrics.get("proto_cos_max", 0.0),
            metrics.get("featmatch", 0.0),
            metrics.get("cycle", 0.0),
            metrics.get("idt", 0.0),
            metrics.get("data_wait_sec", 0.0) * 1000.0,
            metrics.get("step_wall_sec", 0.0) * 1000.0,
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
