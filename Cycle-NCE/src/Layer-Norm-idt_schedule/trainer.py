from __future__ import annotations

import csv
import gc
import inspect
import json
import logging
import math
import os
import shutil
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

try:
    from .losses import AdaCUTObjective
    from .model import build_model_from_config, count_parameters
except ImportError:  # pragma: no cover
    from losses import AdaCUTObjective
    from model import build_model_from_config, count_parameters

logger = logging.getLogger(__name__)

_TRAIN_LOG_COLUMNS = [
    "epoch",
    "loss",
    "swd",
    "repulsive",
    "color",
    "oob",
    "identity",
    "identity_ratio",
    "aent",
    "amax",
    "lr",
    "data_time_sec",
    "transfer_time_sec",
    "fwd_loss_time_sec",
    "backward_time_sec",
    "optimizer_time_sec",
    "step_overhead_time_sec",
    "compute_time_sec",
    "epoch_time_sec",
    "samples_seen",
    "samples_per_sec",
    "compute_samples_per_sec",
]

_SNAPSHOT_SOURCE_FILES = [
    "trainer.py",
    "losses.py",
    "model.py",
    "dataset.py",
    "run.py",
]


def _strip_compile_prefix(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    if any(k.startswith("_orig_mod.") for k in state_dict.keys()):
        return {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}
    return state_dict


class AdaCUTTrainer:
    @staticmethod
    def _unwrap_model(module: torch.nn.Module) -> torch.nn.Module:
        return getattr(module, "_orig_mod", module)

    def _collect_attention_metrics(self) -> Dict[str, torch.Tensor]:
        model = self._unwrap_model(self.model)
        body_blocks = getattr(model, "body_blocks", None)
        if body_blocks is None:
            return {}

        attn_tensors = []
        for block in body_blocks:
            attn = getattr(block, "last_attn", None)
            if torch.is_tensor(attn) and attn.numel() > 0:
                attn_tensors.append(attn)
        if not attn_tensors:
            return {}

        entropies = []
        maxima = []
        for attn in attn_tensors:
            attn_f = attn.detach().float()
            probs = attn_f.clamp_min(1e-8)
            entropy = -(probs * probs.log()).sum(dim=-1)
            entropy = entropy / math.log(float(attn_f.shape[-1]))
            entropies.append(entropy.mean())
            maxima.append(attn_f.amax(dim=-1).mean())

        return {
            "aent": torch.stack(entropies).mean(),
            "amax": torch.stack(maxima).mean(),
        }

    @staticmethod
    def _apply_channels_last_(module: torch.nn.Module) -> None:
        for p in module.parameters():
            if p.ndim == 4:
                p.data = p.data.contiguous(memory_format=torch.channels_last)
        for b in module.buffers():
            if b.ndim == 4:
                b.data = b.data.contiguous(memory_format=torch.channels_last)

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

        cudnn_benchmark_cfg = train_cfg.get("cudnn_benchmark", "auto")
        if isinstance(cudnn_benchmark_cfg, str):
            mode = cudnn_benchmark_cfg.strip().lower()
            if mode == "on":
                torch.backends.cudnn.benchmark = True
            elif mode == "off":
                torch.backends.cudnn.benchmark = False
            else:
                torch.backends.cudnn.benchmark = bool(device.type == "cuda")
        else:
            torch.backends.cudnn.benchmark = bool(cudnn_benchmark_cfg)

        self.channels_last = bool(train_cfg.get("channels_last", False) and device.type == "cuda")
        self.gc_collect_interval = max(0, int(train_cfg.get("gc_collect_interval", 0)))

        grad_ckpt_cfg = bool(train_cfg.get("use_gradient_checkpointing", False))
        self.model = build_model_from_config(model_cfg, use_checkpointing=grad_ckpt_cfg).to(device)
        if self.channels_last:
            self._apply_channels_last_(self.model)

        requested_use_compile = bool(train_cfg.get("use_compile", False))
        self.use_compile = bool(requested_use_compile and device.type == "cuda")
        self.compile_backend = str(train_cfg.get("compile_backend", "inductor"))
        self.compile_mode = str(train_cfg.get("compile_mode", "default"))
        self.compile_fullgraph = bool(train_cfg.get("compile_fullgraph", False))
        if self.use_compile:
            self.model = torch.compile(
                self.model,
                backend=self.compile_backend,
                mode=self.compile_mode,
                fullgraph=self.compile_fullgraph,
            )

        logger.info("Model params: %s", f"{count_parameters(self.model):,}")

        self.use_amp = bool(train_cfg.get("use_amp", False) and device.type == "cuda")
        amp_dtype_cfg = str(train_cfg.get("amp_dtype", "bf16")).lower()
        self.amp_dtype = torch.bfloat16 if amp_dtype_cfg in {"bf16", "bfloat16"} else torch.float16

        requested_fused_adamw = bool(train_cfg.get("fused_adamw", device.type == "cuda"))
        use_fused_adamw = bool(requested_fused_adamw and device.type == "cuda")
        try:
            self.optimizer = torch.optim.AdamW(
                self.model.parameters(),
                lr=float(train_cfg.get("learning_rate", 1e-3)),
                weight_decay=float(train_cfg.get("weight_decay", 1e-4)),
                betas=(0.9, 0.999),
                fused=use_fused_adamw,
            )
        except TypeError:
            self.optimizer = torch.optim.AdamW(
                self.model.parameters(),
                lr=float(train_cfg.get("learning_rate", 1e-3)),
                weight_decay=float(train_cfg.get("weight_decay", 1e-4)),
                betas=(0.9, 0.999),
            )

        self.scheduler = None
        self.scheduler_name = str(train_cfg.get("scheduler", "cosine")).lower()
        self.scheduler_step_mode = "epoch"
        self._onecycle_initialized = False
        self._pending_scheduler_state_dict = None
        if self.scheduler_name == "cosine":
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=max(1, int(train_cfg.get("num_epochs", 100))),
                eta_min=float(train_cfg.get("min_learning_rate", 1e-5)),
            )
        elif self.scheduler_name == "multistep":
            raw_milestones = train_cfg.get("multistep_milestones", [45, 55])
            milestones = sorted({int(v) for v in raw_milestones if int(v) > 0}) if isinstance(raw_milestones, (list, tuple)) else [45, 55]
            if not milestones:
                milestones = [45, 55]
            self.scheduler = torch.optim.lr_scheduler.MultiStepLR(
                self.optimizer,
                milestones=milestones,
                gamma=float(train_cfg.get("multistep_gamma", 0.1)),
            )
        elif self.scheduler_name == "onecycle":
            self.scheduler_step_mode = "batch"
        else:
            self.scheduler_name = "cosine"
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=max(1, int(train_cfg.get("num_epochs", 100))),
                eta_min=float(train_cfg.get("min_learning_rate", 1e-5)),
            )

        self.base_lrs = [float(group.get("lr", 0.0)) for group in self.optimizer.param_groups]
        self.warmup_steps = max(0, int(train_cfg.get("warmup_steps", 0)))
        self.warmup_ratio = max(0.0, float(train_cfg.get("warmup_ratio", 0.0)))
        self.warmup_start_factor = min(max(float(train_cfg.get("warmup_start_factor", 0.0)), 0.0), 1.0)
        self._warmup_ready = self.warmup_steps > 0
        self._warmup_logged = False

        self.loss_fn = AdaCUTObjective(config)

        self.grad_clip_norm = float(train_cfg.get("grad_clip_norm", 1.0))
        self.accumulation_steps = max(1, int(train_cfg.get("accumulation_steps", 1)))
        self.log_interval = max(0, int(train_cfg.get("log_interval", 20)))
        self.use_tqdm = bool(train_cfg.get("use_tqdm", True))
        self.num_epochs = int(train_cfg.get("num_epochs", 100))
        self.save_interval = max(1, int(train_cfg.get("save_interval", 10)))

        self.checkpoint_dir = Path(ckpt_cfg.get("save_dir", "../adacut_ckpt"))
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir = self.checkpoint_dir / "logs"
        self.log_dir.mkdir(parents=True, exist_ok=True)

        with open(self.checkpoint_dir / "config.json", "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2, ensure_ascii=False)

        src_dir = Path(__file__).parent
        for fname in _SNAPSHOT_SOURCE_FILES:
            src = src_dir / fname
            if src.exists():
                shutil.copy2(src, self.checkpoint_dir / fname)

        self.log_file = self.log_dir / f"training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        with open(self.log_file, "w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(_TRAIN_LOG_COLUMNS)

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
        model_state = _strip_compile_prefix(state["model_state_dict"])
        self.model.load_state_dict(model_state, strict=True)

        if "optimizer_state_dict" in state:
            self.optimizer.load_state_dict(state["optimizer_state_dict"])

        if "scheduler_state_dict" in state and state["scheduler_state_dict"] is not None:
            if self.scheduler is not None:
                self.scheduler.load_state_dict(state["scheduler_state_dict"])
            else:
                self._pending_scheduler_state_dict = state["scheduler_state_dict"]

        self.global_step = int(state.get("global_step", 0))
        self.start_epoch = int(state.get("epoch", 0)) + 1
        logger.info("Resumed from %s at epoch=%d global_step=%d", ckpt_path, self.start_epoch, self.global_step)

    def _move_batch(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        out = {}
        for k, v in batch.items():
            if torch.is_tensor(v):
                if v.device == self.device:
                    out[k] = v
                elif self.channels_last and v.is_floating_point() and v.ndim == 4:
                    out[k] = v.to(self.device, non_blocking=True, memory_format=torch.channels_last)
                else:
                    out[k] = v.to(self.device, non_blocking=True)
            else:
                out[k] = v
        return out

    def step_scheduler(self) -> None:
        if self.scheduler is not None and self.scheduler_step_mode == "epoch":
            self.scheduler.step()

    def _init_onecycle_scheduler_if_needed(self, *, total_batches: int) -> None:
        if self.scheduler_name != "onecycle" or self._onecycle_initialized:
            return
        train_cfg = self.config.get("training", {})
        steps_per_epoch = max(1, int(math.ceil(float(total_batches) / float(max(1, self.accumulation_steps)))))
        max_lr_cfg = train_cfg.get("onecycle_max_lr", train_cfg.get("learning_rate", 1e-3))
        max_lr = [float(max_lr_cfg)] * len(self.optimizer.param_groups)
        self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr=max_lr,
            epochs=int(self.num_epochs),
            steps_per_epoch=steps_per_epoch,
            pct_start=float(train_cfg.get("onecycle_pct_start", 0.3)),
            anneal_strategy=str(train_cfg.get("onecycle_anneal_strategy", "cos")).lower(),
            div_factor=float(train_cfg.get("onecycle_div_factor", 25.0)),
            final_div_factor=float(train_cfg.get("onecycle_final_div_factor", 1e4)),
            cycle_momentum=False,
            three_phase=bool(train_cfg.get("onecycle_three_phase", False)),
        )
        self._onecycle_initialized = True
        if self._pending_scheduler_state_dict is not None:
            self.scheduler.load_state_dict(self._pending_scheduler_state_dict)
            self._pending_scheduler_state_dict = None

    def _resolve_warmup_steps(self, *, total_batches: int) -> None:
        if self._warmup_ready or self.warmup_ratio <= 0.0:
            return
        steps_per_epoch = max(1, int(math.ceil(float(total_batches) / float(max(1, self.accumulation_steps)))))
        estimated_total_steps = max(1, int(self.num_epochs) * steps_per_epoch)
        self.warmup_steps = max(1, int(round(estimated_total_steps * self.warmup_ratio)))
        self._warmup_ready = True

    def _apply_warmup_lr(self) -> None:
        if self.warmup_steps <= 0 or self.global_step >= self.warmup_steps:
            return
        progress = float(self.global_step + 1) / float(max(1, self.warmup_steps))
        factor = self.warmup_start_factor + (1.0 - self.warmup_start_factor) * min(max(progress, 0.0), 1.0)
        for base_lr, group in zip(self.base_lrs, self.optimizer.param_groups):
            group["lr"] = float(base_lr) * factor

    def train_epoch(self, dataloader: DataLoader, epoch: int) -> Dict[str, float]:
        self.model.train()
        epoch_start = time.time()

        metric_accum: Dict[str, torch.Tensor] = {}
        num_batches = 0
        data_time_total = 0.0
        transfer_time_total = 0.0
        fwd_loss_time_total = 0.0
        backward_time_total = 0.0
        optimizer_time_total = 0.0
        step_overhead_time_total = 0.0
        compute_time_total = 0.0

        total_steps = len(dataloader)
        self._init_onecycle_scheduler_if_needed(total_batches=total_steps)
        self._resolve_warmup_steps(total_batches=total_steps)
        if self.warmup_steps > 0 and not self._warmup_logged:
            logger.info(
                "LR warmup enabled: warmup_steps=%d start_factor=%.3f base_lr=%s",
                self.warmup_steps,
                self.warmup_start_factor,
                ",".join(f"{v:.3e}" for v in self.base_lrs),
            )
            self._warmup_logged = True

        progress = tqdm(
            dataloader,
            total=total_steps,
            desc=f"Epoch {epoch}/{self.num_epochs}",
            leave=True,
            dynamic_ncols=True,
            mininterval=2.0,
            disable=not self.use_tqdm,
        )

        self.optimizer.zero_grad(set_to_none=True)
        data_wait_start = time.perf_counter()

        for step_idx, raw_batch in enumerate(progress, start=1):
            step_enter = time.perf_counter()
            data_time_total += max(0.0, step_enter - data_wait_start)
            step_compute_start = time.perf_counter()
            transfer_step = 0.0
            fwd_loss_step = 0.0
            backward_step = 0.0
            optimizer_step = 0.0

            t0 = time.perf_counter()
            batch = self._move_batch(raw_batch)
            transfer_step += max(0.0, time.perf_counter() - t0)

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
                compute_kwargs = dict(
                    content=content,
                    target_style=target_style,
                    target_style_id=target_style_id,
                    source_style_id=source_style_id,
                )
                compute_sig = inspect.signature(self.loss_fn.compute)
                if "epoch" in compute_sig.parameters:
                    compute_kwargs["epoch"] = epoch
                if "num_epochs" in compute_sig.parameters:
                    compute_kwargs["num_epochs"] = self.num_epochs
                loss_dict = self.loss_fn.compute(
                    self.model,
                    **compute_kwargs,
                )
                loss = loss_dict["loss"]
            attn_metrics = self._collect_attention_metrics()
            if attn_metrics:
                loss_dict.update(attn_metrics)
            fwd_loss_step += max(0.0, time.perf_counter() - t0)

            loss_to_backward = loss / self.accumulation_steps
            t0 = time.perf_counter()
            loss_to_backward.backward()
            backward_step += max(0.0, time.perf_counter() - t0)

            should_step = (step_idx % self.accumulation_steps == 0)
            if should_step:
                t0 = time.perf_counter()
                if self.grad_clip_norm > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip_norm)
                self._apply_warmup_lr()
                self.optimizer.step()
                if self.scheduler is not None and self.scheduler_step_mode == "batch":
                    self.scheduler.step()
                self.optimizer.zero_grad(set_to_none=True)
                optimizer_step += max(0.0, time.perf_counter() - t0)
                self.global_step += 1
                if self.gc_collect_interval > 0 and (self.global_step % self.gc_collect_interval == 0):
                    gc.collect()

            for k, v in loss_dict.items():
                if v is None or str(k).startswith("_"):
                    continue
                vd = v.detach()
                metric_accum[k] = metric_accum.get(k, 0) + vd
            num_batches += 1

            step_elapsed = max(0.0, time.perf_counter() - step_compute_start)
            step_compute = transfer_step + fwd_loss_step + backward_step + optimizer_step
            step_overhead = max(0.0, step_elapsed - step_compute)
            transfer_time_total += transfer_step
            fwd_loss_time_total += fwd_loss_step
            backward_time_total += backward_step
            optimizer_time_total += optimizer_step
            step_overhead_time_total += step_overhead
            compute_time_total += step_compute

            if self.log_interval > 0 and (step_idx % self.log_interval == 0):
                elapsed = time.time() - epoch_start
                step_per_sec = step_idx / max(elapsed, 1e-6)
                eta = (total_steps - step_idx) / max(step_per_sec, 1e-6)

                def _get_avg(key: str) -> float:
                    if key not in metric_accum:
                        return 0.0
                    return float((metric_accum[key] / num_batches).item())

                progress.set_postfix(
                    loss=f"{_get_avg('loss'):.4f}",
                    swd=f"{_get_avg('swd'):.4f}",
                    rep=f"{_get_avg('repulsive'):.4f}",
                    color=f"{_get_avg('color'):.4f}",
                    idt=f"{_get_avg('identity'):.4f}",
                    idr=f"{_get_avg('identity_ratio'):.2f}",
                    aent=f"{_get_avg('aent'):.3f}",
                    amax=f"{_get_avg('amax'):.3f}",
                    data_ms=f"{(1000.0 * data_time_total / max(step_idx, 1)):.1f}",
                    comp_ms=f"{(1000.0 * compute_time_total / max(step_idx, 1)):.1f}",
                    it_s=f"{step_per_sec:.2f}",
                    eta=f"{eta:.1f}s",
                )

            data_wait_start = time.perf_counter()

            del loss
            del loss_dict
            del batch
            del content
            del target_style
            del target_style_id
            del source_style_id

        progress.close()

        if num_batches > 0 and (num_batches % self.accumulation_steps != 0):
            if self.grad_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip_norm)
            self._apply_warmup_lr()
            self.optimizer.step()
            if self.scheduler is not None and self.scheduler_step_mode == "batch":
                self.scheduler.step()
            self.optimizer.zero_grad(set_to_none=True)
            self.global_step += 1

        epoch_time = time.time() - epoch_start
        lr = float(self.optimizer.param_groups[0]["lr"])
        batch_size = int(getattr(dataloader, "batch_size", 0) or 0)
        samples_seen = int(num_batches * batch_size) if batch_size > 0 else 0
        samples_per_sec = float(samples_seen / max(epoch_time, 1e-6)) if samples_seen > 0 else 0.0
        compute_samples_per_sec = float(samples_seen / max(compute_time_total, 1e-6)) if samples_seen > 0 else 0.0

        metrics: Dict[str, float] = {}
        denom = max(num_batches, 1)
        for k, v in metric_accum.items():
            metrics[k] = float((v / denom).item())

        metrics.setdefault("loss", 0.0)
        metrics.setdefault("swd", 0.0)
        metrics.setdefault("repulsive", 0.0)
        metrics.setdefault("color", 0.0)
        metrics.setdefault("oob", 0.0)
        metrics.setdefault("identity", 0.0)
        metrics.setdefault("identity_ratio", 0.0)
        metrics.setdefault("aent", 0.0)
        metrics.setdefault("amax", 0.0)
        metrics["lr"] = lr
        metrics["data_time_sec"] = data_time_total
        metrics["transfer_time_sec"] = transfer_time_total
        metrics["fwd_loss_time_sec"] = fwd_loss_time_total
        metrics["backward_time_sec"] = backward_time_total
        metrics["optimizer_time_sec"] = optimizer_time_total
        metrics["step_overhead_time_sec"] = step_overhead_time_total
        metrics["compute_time_sec"] = compute_time_total
        metrics["epoch_time_sec"] = epoch_time
        metrics["samples_seen"] = float(samples_seen)
        metrics["samples_per_sec"] = samples_per_sec
        metrics["compute_samples_per_sec"] = compute_samples_per_sec
        return metrics

    def log_epoch(self, epoch: int, metrics: Dict[str, float]) -> None:
        with open(self.log_file, "a", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    int(epoch),
                    float(metrics.get("loss", 0.0)),
                    float(metrics.get("swd", 0.0)),
                    float(metrics.get("repulsive", 0.0)),
                    float(metrics.get("color", 0.0)),
                    float(metrics.get("oob", 0.0)),
                    float(metrics.get("identity", 0.0)),
                    float(metrics.get("identity_ratio", 0.0)),
                    float(metrics.get("aent", 0.0)),
                    float(metrics.get("amax", 0.0)),
                    float(metrics.get("lr", 0.0)),
                    float(metrics.get("data_time_sec", 0.0)),
                    float(metrics.get("transfer_time_sec", 0.0)),
                    float(metrics.get("fwd_loss_time_sec", 0.0)),
                    float(metrics.get("backward_time_sec", 0.0)),
                    float(metrics.get("optimizer_time_sec", 0.0)),
                    float(metrics.get("step_overhead_time_sec", 0.0)),
                    float(metrics.get("compute_time_sec", 0.0)),
                    float(metrics.get("epoch_time_sec", 0.0)),
                    int(float(metrics.get("samples_seen", 0.0))),
                    float(metrics.get("samples_per_sec", 0.0)),
                    float(metrics.get("compute_samples_per_sec", 0.0)),
                ]
            )

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
