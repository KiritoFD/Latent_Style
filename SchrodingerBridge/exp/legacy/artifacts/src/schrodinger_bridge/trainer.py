from __future__ import annotations

import csv
import json
import logging
import math
import shutil
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from .losses import OTFlowMatchingObjective
from .model import build_model_from_config, count_parameters

logger = logging.getLogger(__name__)

_TRAIN_LOG_COLUMNS = [
    "epoch",
    "loss",
    "flow",
    "ot_cost",
    "terminal_swd",
    "plan_entropy",
    "bridge_sigma",
    "identity_ratio",
    "t_mean",
    "velocity_abs",
    "endpoint_abs",
    "lr",
    "data_time_sec",
    "forward_time_sec",
    "backward_time_sec",
    "optimizer_time_sec",
    "compute_time_sec",
    "epoch_time_sec",
    "samples_seen",
    "samples_per_sec",
]

_SNAPSHOT_SOURCE_FILES = [
    "trainer.py",
    "losses.py",
    "model.py",
    "lancet_backbone.py",
    "ot_cost.py",
    "dataset.py",
    "run.py",
    "utils/inference.py",
    "utils/run_evaluation.py",
]


def _strip_compile_prefix(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    if any(k.startswith("_orig_mod.") for k in state_dict.keys()):
        return {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}
    return state_dict


class SBTrainer:
    def __init__(self, config: Dict, device: torch.device, config_path: Optional[str] = None) -> None:
        self.config = config
        self.device = device
        self.config_path = config_path

        train_cfg = config.get("training", {})
        model_cfg = config.get("model", {})
        ckpt_cfg = config.get("checkpoint", {})

        torch.set_float32_matmul_precision("high")
        self.allow_tf32 = bool(train_cfg.get("allow_tf32", True))
        if torch.cuda.is_available():
            torch.backends.cuda.matmul.allow_tf32 = self.allow_tf32
            torch.backends.cudnn.allow_tf32 = self.allow_tf32
            torch.backends.cudnn.benchmark = bool(train_cfg.get("cudnn_benchmark", True))

        self.channels_last = bool(train_cfg.get("channels_last", False) and device.type == "cuda")
        self.use_amp = bool(train_cfg.get("use_amp", False) and device.type == "cuda")
        amp_dtype_cfg = str(train_cfg.get("amp_dtype", "bf16")).lower()
        self.amp_dtype = torch.bfloat16 if amp_dtype_cfg in {"bf16", "bfloat16"} else torch.float16

        self.model = build_model_from_config(
            model_cfg,
            use_checkpointing=bool(train_cfg.get("use_gradient_checkpointing", False)),
        ).to(device)
        if self.channels_last:
            self.model = self.model.to(memory_format=torch.channels_last)

        logger.info("Model params: %s", f"{count_parameters(self.model):,}")

        requested_fused = bool(train_cfg.get("fused_adamw", device.type == "cuda"))
        use_fused = bool(requested_fused and device.type == "cuda")
        try:
            self.optimizer = torch.optim.AdamW(
                self.model.parameters(),
                lr=float(train_cfg.get("learning_rate", 2e-4)),
                weight_decay=float(train_cfg.get("weight_decay", 1e-4)),
                betas=(0.9, 0.999),
                fused=use_fused,
            )
        except TypeError:
            self.optimizer = torch.optim.AdamW(
                self.model.parameters(),
                lr=float(train_cfg.get("learning_rate", 2e-4)),
                weight_decay=float(train_cfg.get("weight_decay", 1e-4)),
                betas=(0.9, 0.999),
            )

        self.scheduler = None
        self.scheduler_name = str(train_cfg.get("scheduler", "cosine")).lower()
        if self.scheduler_name == "multistep":
            milestones = sorted(int(v) for v in train_cfg.get("multistep_milestones", [40, 55]))
            gamma = float(train_cfg.get("multistep_gamma", 0.1))
            self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=milestones, gamma=gamma)
        else:
            self.scheduler_name = "cosine"
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=max(1, int(train_cfg.get("num_epochs", 60))),
                eta_min=float(train_cfg.get("min_learning_rate", 5e-5)),
            )

        self.loss_fn = OTFlowMatchingObjective(config)
        self.grad_clip_norm = float(train_cfg.get("grad_clip_norm", 1.0))
        self.accumulation_steps = max(1, int(train_cfg.get("accumulation_steps", 1)))
        self.log_interval = max(0, int(train_cfg.get("log_interval", 20)))
        self.use_tqdm = bool(train_cfg.get("use_tqdm", True))
        self.num_epochs = int(train_cfg.get("num_epochs", 60))
        self.save_interval = max(1, int(train_cfg.get("save_interval", 10)))

        self.checkpoint_dir = Path(ckpt_cfg.get("save_dir", "./artifacts"))
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir = self.checkpoint_dir / "logs"
        self.log_dir.mkdir(parents=True, exist_ok=True)

        with open(self.checkpoint_dir / "config.json", "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2, ensure_ascii=False)

        pkg_dir = Path(__file__).parent
        snapshot_root = self.checkpoint_dir / "src" / "schrodinger_bridge"
        snapshot_root.mkdir(parents=True, exist_ok=True)
        for fname in _SNAPSHOT_SOURCE_FILES:
            src = pkg_dir / fname
            if src.exists():
                dst = snapshot_root / fname
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, dst)

        self.log_file = self.log_dir / f"training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        with open(self.log_file, "w", encoding="utf-8", newline="") as f:
            csv.writer(f).writerow(_TRAIN_LOG_COLUMNS)

        self.global_step = 0
        self.start_epoch = 1
        self._maybe_resume(str(train_cfg.get("resume_checkpoint", "")))

    def _find_latest_checkpoint(self) -> Optional[Path]:
        ckpts = sorted(self.checkpoint_dir.glob("epoch_*.pt"))
        return ckpts[-1] if ckpts else None

    def _maybe_resume(self, resume_checkpoint: str) -> None:
        if resume_checkpoint:
            ckpt_path = Path(resume_checkpoint)
            if not ckpt_path.is_absolute():
                ckpt_path = (Path.cwd() / ckpt_path).resolve()
        else:
            ckpt_path = self._find_latest_checkpoint()
        if ckpt_path is None or not ckpt_path.exists():
            logger.info("No checkpoint found, start from scratch.")
            return
        state = torch.load(ckpt_path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(_strip_compile_prefix(state["model_state_dict"]), strict=True)
        if "optimizer_state_dict" in state:
            self.optimizer.load_state_dict(state["optimizer_state_dict"])
        if self.scheduler is not None and state.get("scheduler_state_dict") is not None:
            self.scheduler.load_state_dict(state["scheduler_state_dict"])
        self.global_step = int(state.get("global_step", 0))
        self.start_epoch = int(state.get("epoch", 0)) + 1
        logger.info("Resumed from %s at epoch=%d global_step=%d", ckpt_path, self.start_epoch, self.global_step)

    def _move_batch(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        out: Dict[str, torch.Tensor] = {}
        for key, value in batch.items():
            if torch.is_tensor(value):
                if self.channels_last and value.is_floating_point() and value.ndim == 4:
                    out[key] = value.to(self.device, non_blocking=True, memory_format=torch.channels_last)
                else:
                    out[key] = value.to(self.device, non_blocking=True)
            else:
                out[key] = value
        return out

    def step_scheduler(self) -> None:
        if self.scheduler is not None:
            self.scheduler.step()

    def train_epoch(self, dataloader: DataLoader, epoch: int) -> Dict[str, float]:
        self.model.train()
        epoch_start = time.time()
        metric_accum: Dict[str, torch.Tensor] = {}
        num_batches = 0
        data_time_total = 0.0
        forward_time_total = 0.0
        backward_time_total = 0.0
        optimizer_time_total = 0.0
        compute_time_total = 0.0

        progress = tqdm(
            dataloader,
            total=len(dataloader),
            desc=f"Epoch {epoch}/{self.num_epochs}",
            dynamic_ncols=True,
            leave=True,
            disable=not self.use_tqdm,
        )

        self.optimizer.zero_grad(set_to_none=True)
        data_wait_start = time.perf_counter()

        def _avg(name: str) -> float:
            if name not in metric_accum or num_batches <= 0:
                return 0.0
            return float((metric_accum[name] / num_batches).item())

        for step_idx, raw_batch in enumerate(progress, start=1):
            step_enter = time.perf_counter()
            data_time_total += max(0.0, step_enter - data_wait_start)

            batch = self._move_batch(raw_batch)
            content = batch["content"]
            target_style = batch["target_style"]
            target_style_id = batch["target_style_id"]
            source_style_id = batch.get("source_style_id")

            t0 = time.perf_counter()
            if self.device.type == "cuda":
                autocast_ctx = torch.amp.autocast("cuda", enabled=self.use_amp, dtype=self.amp_dtype)
            else:
                autocast_ctx = torch.autocast("cpu", enabled=False)
            with autocast_ctx:
                loss_dict = self.loss_fn.compute(
                    self.model,
                    content=content,
                    target_style=target_style,
                    target_style_id=target_style_id,
                    source_style_id=source_style_id,
                )
                loss = loss_dict["loss"]
            forward_time_total += max(0.0, time.perf_counter() - t0)

            t0 = time.perf_counter()
            (loss / self.accumulation_steps).backward()
            backward_time_total += max(0.0, time.perf_counter() - t0)

            should_step = (step_idx % self.accumulation_steps == 0)
            if should_step:
                t0 = time.perf_counter()
                if self.grad_clip_norm > 0.0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip_norm)
                self.optimizer.step()
                self.optimizer.zero_grad(set_to_none=True)
                optimizer_time_total += max(0.0, time.perf_counter() - t0)
                self.global_step += 1

            for key, value in loss_dict.items():
                if value is None:
                    continue
                metric_accum[key] = metric_accum.get(key, 0) + value.detach()
            num_batches += 1

            compute_time_total = forward_time_total + backward_time_total + optimizer_time_total
            if self.use_tqdm:
                progress.set_postfix(
                    loss=f"{_avg('loss'):.4f}",
                    flow=f"{_avg('flow'):.4f}",
                    ot=f"{_avg('ot_cost'):.4f}",
                    tswd=f"{_avg('terminal_swd'):.4f}",
                    t=f"{_avg('t_mean'):.3f}",
                )

            data_wait_start = time.perf_counter()

            del loss
            del loss_dict
            del batch

        progress.close()

        if num_batches > 0 and (num_batches % self.accumulation_steps != 0):
            if self.grad_clip_norm > 0.0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip_norm)
            self.optimizer.step()
            self.optimizer.zero_grad(set_to_none=True)
            self.global_step += 1

        epoch_time = time.time() - epoch_start
        batch_size = int(getattr(dataloader, "batch_size", 0) or 0)
        samples_seen = int(num_batches * batch_size)
        samples_per_sec = float(samples_seen / max(epoch_time, 1e-6)) if samples_seen > 0 else 0.0

        metrics: Dict[str, float] = {}
        denom = max(num_batches, 1)
        for key, value in metric_accum.items():
            metrics[key] = float((value / denom).item())
        metrics.setdefault("loss", 0.0)
        metrics.setdefault("flow", 0.0)
        metrics.setdefault("ot_cost", 0.0)
        metrics.setdefault("terminal_swd", 0.0)
        metrics.setdefault("plan_entropy", 0.0)
        metrics.setdefault("bridge_sigma", 0.0)
        metrics.setdefault("identity_ratio", 0.0)
        metrics.setdefault("t_mean", 0.0)
        metrics.setdefault("velocity_abs", 0.0)
        metrics.setdefault("endpoint_abs", 0.0)
        metrics["lr"] = float(self.optimizer.param_groups[0]["lr"])
        metrics["data_time_sec"] = data_time_total
        metrics["forward_time_sec"] = forward_time_total
        metrics["backward_time_sec"] = backward_time_total
        metrics["optimizer_time_sec"] = optimizer_time_total
        metrics["compute_time_sec"] = compute_time_total
        metrics["epoch_time_sec"] = epoch_time
        metrics["samples_seen"] = float(samples_seen)
        metrics["samples_per_sec"] = samples_per_sec
        return metrics

    def log_epoch(self, epoch: int, metrics: Dict[str, float]) -> None:
        row = [
            int(epoch),
            float(metrics.get("loss", 0.0)),
            float(metrics.get("flow", 0.0)),
            float(metrics.get("ot_cost", 0.0)),
            float(metrics.get("terminal_swd", 0.0)),
            float(metrics.get("plan_entropy", 0.0)),
            float(metrics.get("bridge_sigma", 0.0)),
            float(metrics.get("identity_ratio", 0.0)),
            float(metrics.get("t_mean", 0.0)),
            float(metrics.get("velocity_abs", 0.0)),
            float(metrics.get("endpoint_abs", 0.0)),
            float(metrics.get("lr", 0.0)),
            float(metrics.get("data_time_sec", 0.0)),
            float(metrics.get("forward_time_sec", 0.0)),
            float(metrics.get("backward_time_sec", 0.0)),
            float(metrics.get("optimizer_time_sec", 0.0)),
            float(metrics.get("compute_time_sec", 0.0)),
            float(metrics.get("epoch_time_sec", 0.0)),
            int(float(metrics.get("samples_seen", 0.0))),
            float(metrics.get("samples_per_sec", 0.0)),
        ]
        with open(self.log_file, "a", encoding="utf-8", newline="") as f:
            csv.writer(f).writerow(row)

    def save_checkpoint(self, epoch: int, metrics: Dict[str, float]) -> Path:
        path = self.checkpoint_dir / f"epoch_{epoch:04d}.pt"
        payload = {
            "epoch": int(epoch),
            "global_step": int(self.global_step),
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict() if self.scheduler is not None else None,
            "config": self.config,
            "metrics": metrics,
        }
        torch.save(payload, path)
        logger.info("Saved checkpoint: %s", path)
        return path
