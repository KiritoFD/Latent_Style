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

from losses import OTFlowMatchingObjective
from model import build_model_from_config, count_parameters

logger = logging.getLogger(__name__)

_TRAIN_LOG_COLUMNS = [
    "epoch",
    "loss",
    "flow",
    "kinetic_energy",
    "low_freq_anchor",
    "ot_cost",
    "terminal_swd",
    "color",
    "patch_nce",
    "cycle",
    "repulsive",
    "semantic_attn_mean",
    "semantic_k_abs",
    "plan_entropy",
    "bridge_sigma",
    "identity_ratio",
    "t_mean",
    "velocity_abs",
    "endpoint_abs",
    "velocity_max",
    "endpoint_max",
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
        self.train_cfg = train_cfg
        self.distill_cfg = dict(train_cfg.get("distill", {}) or {})
        self.distill_enabled = bool(self.distill_cfg.get("enabled", False))
        self.teacher_model = None

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

        self.optimizer = self._build_optimizer(self.model.parameters())

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
        self.numeric_debug = bool(train_cfg.get("numeric_debug", False))
        self.numeric_debug_interval = max(1, int(train_cfg.get("numeric_debug_interval", 10)))
        self.numeric_debug_halt_on_nonfinite = bool(train_cfg.get("numeric_debug_halt_on_nonfinite", True))
        self.numeric_debug_dump_limit = max(1, int(train_cfg.get("numeric_debug_dump_limit", 200)))
        self.numeric_debug_events = 0

        self.checkpoint_dir = Path(ckpt_cfg.get("save_dir", "./artifacts"))
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir = self.checkpoint_dir / "logs"
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.numeric_debug_file = self.checkpoint_dir / "numeric_debug.jsonl"

        with open(self.checkpoint_dir / "config.json", "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2, ensure_ascii=False)

        pkg_dir = Path(__file__).parent
        snapshot_root = self.checkpoint_dir / "src"
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
        self._configure_distillation()

    def _tensor_stats(self, value: torch.Tensor | None) -> Dict[str, float]:
        if value is None:
            return {
                "is_present": 0.0,
                "finite_ratio": 1.0,
                "mean_abs": 0.0,
                "max_abs": 0.0,
                "mean": 0.0,
            }
        x = value.detach().float()
        finite = torch.isfinite(x)
        finite_ratio = float(finite.float().mean().item()) if x.numel() > 0 else 1.0
        safe = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        return {
            "is_present": 1.0,
            "finite_ratio": finite_ratio,
            "mean_abs": float(safe.abs().mean().item()) if safe.numel() > 0 else 0.0,
            "max_abs": float(safe.abs().amax().item()) if safe.numel() > 0 else 0.0,
            "mean": float(safe.mean().item()) if safe.numel() > 0 else 0.0,
        }

    def _grad_stats(self) -> Dict[str, float | str | None]:
        grad_max = 0.0
        grad_mean = 0.0
        counted = 0
        first_nonfinite_name = None
        first_nonfinite_ratio = 1.0
        max_name = None
        for name, param in self.model.named_parameters():
            grad = param.grad
            if grad is None:
                continue
            g = grad.detach().float()
            finite = torch.isfinite(g)
            finite_ratio = float(finite.float().mean().item()) if g.numel() > 0 else 1.0
            safe = torch.nan_to_num(g, nan=0.0, posinf=0.0, neginf=0.0)
            gmax = float(safe.abs().amax().item()) if safe.numel() > 0 else 0.0
            grad_mean += float(safe.abs().mean().item()) if safe.numel() > 0 else 0.0
            counted += 1
            if gmax > grad_max:
                grad_max = gmax
                max_name = name
            if first_nonfinite_name is None and finite_ratio < 1.0:
                first_nonfinite_name = name
                first_nonfinite_ratio = finite_ratio
        return {
            "grad_abs_max": grad_max,
            "grad_abs_mean": (grad_mean / counted) if counted > 0 else 0.0,
            "grad_abs_max_name": max_name,
            "first_nonfinite_grad_name": first_nonfinite_name,
            "first_nonfinite_grad_ratio": first_nonfinite_ratio,
        }

    def _write_numeric_debug(
        self,
        *,
        epoch: int,
        step: int,
        stage: str,
        loss_dict: Dict[str, torch.Tensor],
        target_style_id: torch.Tensor,
        source_style_id: torch.Tensor | None,
        extra: Optional[Dict[str, object]] = None,
    ) -> None:
        if not self.numeric_debug or self.numeric_debug_events >= self.numeric_debug_dump_limit:
            return
        payload: Dict[str, object] = {
            "epoch": int(epoch),
            "step": int(step),
            "stage": stage,
            "global_step": int(self.global_step),
            "loss": float(torch.nan_to_num(loss_dict["loss"].detach().float(), nan=0.0, posinf=0.0, neginf=0.0).item()),
            "loss_is_finite": bool(torch.isfinite(loss_dict["loss"].detach()).item()),
            "metrics": {
                key: float(torch.nan_to_num(value.detach().float(), nan=0.0, posinf=0.0, neginf=0.0).item())
                for key, value in loss_dict.items()
                if torch.is_tensor(value) and value.ndim == 0
            },
            "target_style_ids": [int(v) for v in target_style_id.detach().cpu().tolist()],
            "source_style_ids": [int(v) for v in source_style_id.detach().cpu().tolist()] if source_style_id is not None else None,
            "semantic_attn": self._tensor_stats(getattr(self.model, "last_semantic_attn", None)),
            "semantic_k": self._tensor_stats(getattr(self.model, "last_semantic_k", None)),
        }
        if extra:
            payload["extra"] = extra
        self.numeric_debug_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self.numeric_debug_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")
        self.numeric_debug_events += 1

    def _build_optimizer(self, params) -> torch.optim.Optimizer:
        requested_fused = bool(self.train_cfg.get("fused_adamw", self.device.type == "cuda"))
        use_fused = bool(requested_fused and self.device.type == "cuda")
        try:
            return torch.optim.AdamW(
                params,
                lr=float(self.train_cfg.get("learning_rate", 2e-4)),
                weight_decay=float(self.train_cfg.get("weight_decay", 1e-4)),
                betas=(0.9, 0.999),
                fused=use_fused,
            )
        except TypeError:
            return torch.optim.AdamW(
                params,
                lr=float(self.train_cfg.get("learning_rate", 2e-4)),
                weight_decay=float(self.train_cfg.get("weight_decay", 1e-4)),
                betas=(0.9, 0.999),
            )

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

    def _reset_trainable_style_params(self, mode: str) -> None:
        with torch.no_grad():
            if mode in {"style_emb_only", "style_branch"} and hasattr(self.model, "style_emb"):
                torch.nn.init.normal_(self.model.style_emb.weight, mean=0.0, std=0.02)
            if mode == "style_branch" and hasattr(self.model, "style_spatial_id_16"):
                torch.nn.init.normal_(self.model.style_spatial_id_16, mean=0.0, std=0.02)

    def _configure_distillation(self) -> None:
        if not self.distill_enabled:
            return
        teacher_ckpt_raw = str(self.distill_cfg.get("teacher_checkpoint", "")).strip()
        if not teacher_ckpt_raw:
            raise ValueError("training.distill.teacher_checkpoint is required when distillation is enabled.")
        teacher_ckpt = Path(teacher_ckpt_raw)
        if not teacher_ckpt.is_absolute():
            teacher_ckpt = (Path.cwd() / teacher_ckpt).resolve()
        if not teacher_ckpt.exists():
            raise FileNotFoundError(f"Distillation teacher checkpoint not found: {teacher_ckpt}")

        state = torch.load(teacher_ckpt, map_location=self.device, weights_only=False)
        teacher_state = _strip_compile_prefix(state["model_state_dict"])
        if not str(self.train_cfg.get("resume_checkpoint", "")).strip():
            self.model.load_state_dict(teacher_state, strict=True)

        teacher = build_model_from_config(
            self.config.get("model", {}),
            use_checkpointing=False,
        ).to(self.device)
        if self.channels_last:
            teacher = teacher.to(memory_format=torch.channels_last)
        teacher.load_state_dict(teacher_state, strict=True)
        teacher.eval()
        for param in teacher.parameters():
            param.requires_grad_(False)
        self.teacher_model = teacher

        mode = str(self.distill_cfg.get("mode", "style_emb_only")).strip().lower()
        if mode not in {"style_emb_only", "style_branch"}:
            raise ValueError(f"Unsupported distill mode: {mode}")

        for _, param in self.model.named_parameters():
            param.requires_grad_(False)

        trainable_names: list[str] = []
        if mode in {"style_emb_only", "style_branch"}:
            self.model.style_emb.weight.requires_grad_(True)
            trainable_names.append("style_emb.weight")
        if mode == "style_branch":
            self.model.style_spatial_id_16.requires_grad_(True)
            trainable_names.append("style_spatial_id_16")

        if bool(self.distill_cfg.get("reinit_trainable", True)):
            self._reset_trainable_style_params(mode)

        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        if not trainable_params:
            raise RuntimeError("Distillation enabled but no trainable parameters were selected.")
        self.optimizer = self._build_optimizer(trainable_params)
        if self.scheduler is not None:
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=max(1, int(self.train_cfg.get("num_epochs", 60))),
                eta_min=float(self.train_cfg.get("min_learning_rate", 5e-5)),
            )
        logger.info("Distill mode=%s | trainable=%s", mode, ", ".join(trainable_names))

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
                if self.distill_enabled and self.teacher_model is not None:
                    loss_dict = self.loss_fn.compute_distill(
                        self.model,
                        self.teacher_model,
                        content=content,
                        target_style=target_style,
                        target_style_id=target_style_id,
                        source_style_id=source_style_id,
                    )
                else:
                    loss_dict = self.loss_fn.compute(
                        self.model,
                        content=content,
                        target_style=target_style,
                        target_style_id=target_style_id,
                        source_style_id=source_style_id,
                    )
                loss = loss_dict["loss"]
            forward_time_total += max(0.0, time.perf_counter() - t0)
            if self.numeric_debug and (step_idx == 1 or step_idx % self.numeric_debug_interval == 0 or not torch.isfinite(loss.detach()).item()):
                self._write_numeric_debug(
                    epoch=epoch,
                    step=step_idx,
                    stage="forward",
                    loss_dict=loss_dict,
                    target_style_id=target_style_id,
                    source_style_id=source_style_id,
                )
            if not torch.isfinite(loss.detach()).item():
                msg = f"Non-finite loss detected at epoch={epoch} step={step_idx}"
                logger.error(msg)
                if self.numeric_debug_halt_on_nonfinite:
                    raise FloatingPointError(msg)

            t0 = time.perf_counter()
            (loss / self.accumulation_steps).backward()
            backward_time_total += max(0.0, time.perf_counter() - t0)
            grad_report = self._grad_stats()
            has_nonfinite_grad = bool(grad_report["first_nonfinite_grad_name"] is not None)
            if self.numeric_debug and (step_idx == 1 or step_idx % self.numeric_debug_interval == 0 or has_nonfinite_grad):
                self._write_numeric_debug(
                    epoch=epoch,
                    step=step_idx,
                    stage="backward",
                    loss_dict=loss_dict,
                    target_style_id=target_style_id,
                    source_style_id=source_style_id,
                    extra=grad_report,
                )
            if has_nonfinite_grad:
                msg = (
                    f"Non-finite gradient detected at epoch={epoch} step={step_idx} "
                    f"param={grad_report['first_nonfinite_grad_name']}"
                )
                logger.error(msg)
                if self.numeric_debug_halt_on_nonfinite:
                    raise FloatingPointError(msg)

            should_step = (step_idx % self.accumulation_steps == 0)
            if should_step:
                t0 = time.perf_counter()
                total_grad_norm = None
                if self.grad_clip_norm > 0.0:
                    total_grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip_norm)
                self.optimizer.step()
                self.optimizer.zero_grad(set_to_none=True)
                optimizer_time_total += max(0.0, time.perf_counter() - t0)
                self.global_step += 1
                if self.numeric_debug and (step_idx == 1 or step_idx % self.numeric_debug_interval == 0):
                    extra = {"clipped_grad_norm": float(total_grad_norm.item()) if total_grad_norm is not None else 0.0}
                    self._write_numeric_debug(
                        epoch=epoch,
                        step=step_idx,
                        stage="optimizer",
                        loss_dict=loss_dict,
                        target_style_id=target_style_id,
                        source_style_id=source_style_id,
                        extra=extra,
                    )

            for key, value in loss_dict.items():
                if value is None:
                    continue
                metric_accum[key] = metric_accum.get(key, 0) + value.detach()
            num_batches += 1

            compute_time_total = forward_time_total + backward_time_total + optimizer_time_total
            if self.use_tqdm:
                progress.set_postfix(
                    loss=f"{_avg('loss'):.4f}",
                    flow=f"{_avg('flow'):.4f}" if not self.distill_enabled else f"{_avg('distill_velocity'):.4f}",
                    ot=f"{_avg('ot_cost'):.4f}",
                    tswd=f"{_avg('terminal_swd'):.4f}" if not self.distill_enabled else f"{_avg('distill_endpoint'):.4f}",
                    low=f"{_avg('low_freq_anchor'):.4f}",
                    color=f"{_avg('color'):.4f}",
                    nce=f"{_avg('patch_nce'):.4f}",
                    cyc=f"{_avg('cycle'):.4f}",
                    rep=f"{_avg('repulsive'):.4f}",
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
        metrics.setdefault("kinetic_energy", 0.0)
        metrics.setdefault("distill_velocity", 0.0)
        metrics.setdefault("distill_endpoint", 0.0)
        metrics.setdefault("ot_cost", 0.0)
        metrics.setdefault("terminal_swd", 0.0)
        metrics.setdefault("color", 0.0)
        metrics.setdefault("repulsive", 0.0)
        metrics.setdefault("plan_entropy", 0.0)
        metrics.setdefault("bridge_sigma", 0.0)
        metrics.setdefault("identity_ratio", 0.0)
        metrics.setdefault("t_mean", 0.0)
        metrics.setdefault("velocity_abs", 0.0)
        metrics.setdefault("endpoint_abs", 0.0)
        metrics.setdefault("velocity_max", 0.0)
        metrics.setdefault("endpoint_max", 0.0)
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
        self.log_dir.mkdir(parents=True, exist_ok=True)
        row = [
            int(epoch),
            float(metrics.get("loss", 0.0)),
            float(metrics.get("flow", 0.0)),
            float(metrics.get("kinetic_energy", 0.0)),
            float(metrics.get("ot_cost", 0.0)),
            float(metrics.get("terminal_swd", 0.0)),
            float(metrics.get("color", 0.0)),
            float(metrics.get("repulsive", 0.0)),
            float(metrics.get("plan_entropy", 0.0)),
            float(metrics.get("bridge_sigma", 0.0)),
            float(metrics.get("identity_ratio", 0.0)),
            float(metrics.get("t_mean", 0.0)),
            float(metrics.get("velocity_abs", 0.0)),
            float(metrics.get("endpoint_abs", 0.0)),
            float(metrics.get("velocity_max", 0.0)),
            float(metrics.get("endpoint_max", 0.0)),
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
        self.log_file.parent.mkdir(parents=True, exist_ok=True)
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
