from __future__ import annotations

import json
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from config_schema import ExperimentConfig, compact_runtime_config
from losses import OTFlowMatchingObjective
from model import build_model_from_config, count_parameters
from utils.training import (
    append_training_log,
    build_adamw,
    initialize_training_log,
    strip_compile_prefix,
    write_config_and_source_snapshot,
)

logger = logging.getLogger(__name__)


class SBTrainer:
    def __init__(self, config: ExperimentConfig, device: torch.device, config_path: Optional[str] = None) -> None:
        self.config = config
        self.serialized_config = compact_runtime_config(config)
        self.device = device
        self.config_path = config_path

        train_cfg = config.training.to_dict()
        model_cfg = config.model
        ckpt_cfg = config.checkpoint
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
        potential_params = self.loss_fn.potential_parameters()
        self.potential_optimizer = (
            torch.optim.AdamW(
                potential_params,
                lr=float(config.bridge.kantorovich_lr),
                weight_decay=0.0,
            )
            if potential_params
            else None
        )
        self.grad_clip_norm = float(train_cfg.get("grad_clip_norm", 1.0))
        self.accumulation_steps = max(1, int(train_cfg.get("accumulation_steps", 1)))
        self.log_interval = max(0, int(train_cfg.get("log_interval", 20)))
        self.use_tqdm = bool(train_cfg.get("use_tqdm", True))
        self.num_epochs = int(train_cfg.get("num_epochs", 60))
        self.save_interval = max(1, int(train_cfg.get("save_interval", 10)))
        self.max_train_batches_per_epoch = max(0, int(train_cfg.get("max_train_batches_per_epoch", 0)))
        self.numeric_debug = bool(train_cfg.get("numeric_debug", False))
        self.numeric_debug_interval = max(1, int(train_cfg.get("numeric_debug_interval", 10)))
        self.numeric_debug_halt_on_nonfinite = bool(train_cfg.get("numeric_debug_halt_on_nonfinite", True))
        self.numeric_debug_dump_limit = max(1, int(train_cfg.get("numeric_debug_dump_limit", 200)))
        self.numeric_debug_events = 0

        self.checkpoint_dir = Path(ckpt_cfg.save_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir = self.checkpoint_dir / "logs"
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.numeric_debug_file = self.checkpoint_dir / "numeric_debug.jsonl"

        write_config_and_source_snapshot(
            checkpoint_dir=self.checkpoint_dir,
            serialized_config=self.serialized_config,
            package_dir=Path(__file__).parent,
            config_path=Path(self.config_path).resolve() if self.config_path else None,
        )

        self.log_file = self.log_dir / f"training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        initialize_training_log(self.log_file)

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
        if x.numel() == 0 or bool(finite.all().item()):
            finite_ratio = 1.0
        else:
            finite_ratio = float(finite.float().mean().item())
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
            all_finite = bool(finite.all().item()) if g.numel() > 0 else True
            finite_ratio = 1.0 if all_finite else float(finite.float().mean().item())
            safe = torch.nan_to_num(g, nan=0.0, posinf=0.0, neginf=0.0)
            gmax = float(safe.abs().amax().item()) if safe.numel() > 0 else 0.0
            grad_mean += float(safe.abs().mean().item()) if safe.numel() > 0 else 0.0
            counted += 1
            if gmax > grad_max:
                grad_max = gmax
                max_name = name
            if first_nonfinite_name is None and not all_finite:
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
        carrier_debug = dict(getattr(self.model, "carrier_debug", {}) or {})
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
            "carrier_debug": {
                key: self._tensor_stats(value)
                for key, value in carrier_debug.items()
            },
            "carrier_debug_by_target_style": {
                key: self._tensor_stats_by_style(value, target_style_id)
                for key, value in carrier_debug.items()
            },
        }
        if extra:
            payload["extra"] = extra
        self.numeric_debug_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self.numeric_debug_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")
        self.numeric_debug_events += 1

    def _tensor_stats_by_style(self, tensor: torch.Tensor | None, style_ids: torch.Tensor) -> Dict[str, Dict[str, object]]:
        if tensor is None or not torch.is_tensor(tensor):
            return {}
        if tensor.ndim == 0 or int(tensor.shape[0]) != int(style_ids.shape[0]):
            return {}
        out: Dict[str, Dict[str, float | int]] = {}
        ids = style_ids.detach().long().cpu()
        flat = tensor.detach().float().cpu().reshape(int(tensor.shape[0]), -1)
        for sid in torch.unique(ids, sorted=True).tolist():
            mask = ids == int(sid)
            if not bool(mask.any()):
                continue
            values = torch.nan_to_num(flat[mask], nan=0.0, posinf=0.0, neginf=0.0)
            if values.numel() == 0:
                continue
            out[str(int(sid))] = {
                "count": int(mask.sum().item()),
                "mean": float(values.mean().item()),
                "std": float(values.std(unbiased=False).item()),
                "abs_mean": float(values.abs().mean().item()),
                "max": float(values.max().item()),
                "min": float(values.min().item()),
            }
            # Keep tiny tokenizer/control vectors interpretable for later
            # field-level diagnosis. Dense spatial tensors stay summarized.
            if values.ndim == 2 and values.shape[1] <= 16:
                component_mean = values.mean(dim=0)
                component_abs_mean = values.abs().mean(dim=0)
                out[str(int(sid))]["component_mean"] = [
                    float(v) for v in component_mean.tolist()
                ]
                out[str(int(sid))]["component_abs_mean"] = [
                    float(v) for v in component_abs_mean.tolist()
                ]
        return out

    def _build_optimizer(self, params) -> torch.optim.Optimizer:
        return build_adamw(params, self.train_cfg, self.device)

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
        self.model.load_state_dict(strip_compile_prefix(state["model_state_dict"]), strict=True)
        if "optimizer_state_dict" in state:
            self.optimizer.load_state_dict(state["optimizer_state_dict"])
        if self.potential_optimizer is not None and state.get("potential_optimizer_state_dict") is not None:
            self.potential_optimizer.load_state_dict(state["potential_optimizer_state_dict"])
        if self.loss_fn.kantorovich_potential is not None and state.get("kantorovich_potential_state_dict") is not None:
            self.loss_fn.kantorovich_potential.load_state_dict(state["kantorovich_potential_state_dict"])
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
        teacher_state = strip_compile_prefix(state["model_state_dict"])
        if not str(self.train_cfg.get("resume_checkpoint", "")).strip():
            self.model.load_state_dict(teacher_state, strict=True)

        teacher = build_model_from_config(
            self.config.model,
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

        progress_total = len(dataloader)
        if self.max_train_batches_per_epoch > 0:
            progress_total = min(progress_total, self.max_train_batches_per_epoch)
        progress = tqdm(
            dataloader,
            total=progress_total,
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
            if self.max_train_batches_per_epoch > 0 and step_idx > self.max_train_batches_per_epoch:
                break
            step_enter = time.perf_counter()
            data_time_total += max(0.0, step_enter - data_wait_start)

            batch = self._move_batch(raw_batch)
            content = batch["content"]
            target_style = batch["target_style"]
            target_style_id = batch["target_style_id"]
            source_style_id = batch.get("source_style_id")
            kantorovich_critic_value = 0.0

            if self.potential_optimizer is not None and not self.distill_enabled:
                for _ in range(max(1, int(self.config.bridge.kantorovich_steps))):
                    self.potential_optimizer.zero_grad(set_to_none=True)
                    critic_loss = self.loss_fn.compute_kantorovich_critic(
                        self.model,
                        content=content,
                        target_style=target_style,
                        target_style_id=target_style_id,
                    )
                    if critic_loss is None:
                        break
                    critic_loss.backward()
                    self.potential_optimizer.step()
                    kantorovich_critic_value = float(critic_loss.detach().float().item())

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
                if self.potential_optimizer is not None:
                    loss_dict["kantorovich_critic"] = content.new_tensor(kantorovich_critic_value)
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
                    kin=f"{_avg('kinetic_energy'):.4f}",
                    curv=f"{_avg('curvature'):.4f}",
                    ot=f"{_avg('ot_cost'):.4f}",
                    tswd=f"{_avg('terminal_swd'):.4f}" if not self.distill_enabled else f"{_avg('distill_endpoint'):.4f}",
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
        metrics.setdefault("anisotropic_kinetic", 0.0)
        metrics.setdefault("stokes_viscous", 0.0)
        metrics.setdefault("phase_separation", 0.0)
        metrics.setdefault("fourier_phase_lock", 0.0)
        metrics.setdefault("head_tax", 0.0)
        metrics.setdefault("curvature", 0.0)
        metrics.setdefault("distill_velocity", 0.0)
        metrics.setdefault("distill_endpoint", 0.0)
        metrics.setdefault("ot_cost", 0.0)
        metrics.setdefault("terminal_swd", 0.0)
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
        append_training_log(self.log_file, metrics, epoch)

    def save_checkpoint(self, epoch: int, metrics: Dict[str, float]) -> Path:
        path = self.checkpoint_dir / f"epoch_{epoch:04d}.pt"
        payload = {
            "epoch": int(epoch),
            "global_step": int(self.global_step),
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "potential_optimizer_state_dict": self.potential_optimizer.state_dict() if self.potential_optimizer is not None else None,
            "kantorovich_potential_state_dict": (
                self.loss_fn.kantorovich_potential.state_dict() if self.loss_fn.kantorovich_potential is not None else None
            ),
            "scheduler_state_dict": self.scheduler.state_dict() if self.scheduler is not None else None,
            "config": self.serialized_config,
            "metrics": metrics,
        }
        torch.save(payload, path)
        logger.info("Saved checkpoint: %s", path)
        return path
