from __future__ import annotations

import csv
import json
import shutil
import logging
import os
import subprocess
import sys
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

_COMPILE_BACKEND = "inductor"
_COMPILE_MODE = "default"
_COMPILE_FULLGRAPH = False


def _strip_compile_prefix(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    if any(k.startswith("_orig_mod.") for k in state_dict.keys()):
        return {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}
    return state_dict


class AdaCUTTrainer:
    def __init__(self, config: Dict, device: torch.device, config_path: Optional[str] = None) -> None:
        self.config = config
        self.device = device
        self.config_path = config_path

        train_cfg = config.get("training", {})
        torch.set_float32_matmul_precision("high")
        self.allow_tf32 = bool(train_cfg.get("allow_tf32", True))
        if torch.cuda.is_available():
            torch.backends.cuda.matmul.allow_tf32 = self.allow_tf32
            torch.backends.cudnn.allow_tf32 = self.allow_tf32
            torch.backends.cudnn.benchmark = True

        self.channels_last = bool(train_cfg.get("channels_last", True) and device.type == "cuda")
        self.skip_oom_batches = bool(train_cfg.get("skip_oom_batches", True))
        self.empty_cache_interval = max(0, int(train_cfg.get("empty_cache_interval", 0)))
        self.log_vram_interval = max(0, int(train_cfg.get("log_vram_interval", 0)))

        model_cfg = config.get("model", {})
        self.model = build_model_from_config(
            model_cfg,
            use_checkpointing=bool(train_cfg.get("use_gradient_checkpointing", False)),
        )
        if self.channels_last:
            self.model = self.model.to(device, memory_format=torch.channels_last)
        else:
            self.model = self.model.to(device)

        ckpt_cfg = config.get("checkpoint", {})
        self.checkpoint_dir = Path(ckpt_cfg.get("save_dir", "../adacut_ckpt"))
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir = self.checkpoint_dir / "logs"
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.full_eval_root = self.checkpoint_dir / "full_eval"
        self.full_eval_root.mkdir(parents=True, exist_ok=True)

        self.use_compile = bool(train_cfg.get("use_compile", False))
        self.compile_backend = _COMPILE_BACKEND
        self.compile_mode = _COMPILE_MODE
        self.compile_fullgraph = _COMPILE_FULLGRAPH
        self.compile_cache_dir = (self.checkpoint_dir / "torch_compile_cache").resolve()
        if self.use_compile:
            try:
                (self.compile_cache_dir / "inductor").mkdir(parents=True, exist_ok=True)
                (self.compile_cache_dir / "triton").mkdir(parents=True, exist_ok=True)
                os.environ["TORCHINDUCTOR_CACHE_DIR"] = str((self.compile_cache_dir / "inductor"))
                os.environ["TRITON_CACHE_DIR"] = str((self.compile_cache_dir / "triton"))
                os.environ.setdefault("TORCHINDUCTOR_CUDAGRAPHS", "0")
                try:
                    import torch._dynamo as _dynamo  # type: ignore[attr-defined]

                    _dynamo.config.capture_scalar_outputs = True
                    _dynamo.config.suppress_errors = True
                except Exception:  # pragma: no cover
                    pass
                try:
                    import torch._inductor.config as _inductor_config  # type: ignore[attr-defined]

                    if hasattr(_inductor_config, "fx_graph_cache"):
                        _inductor_config.fx_graph_cache = True
                except Exception:  # pragma: no cover
                    pass
                self.model = torch.compile(
                    self.model,
                    backend=self.compile_backend,
                    mode=self.compile_mode,
                    fullgraph=self.compile_fullgraph,
                )
                logger.info(
                    "torch.compile enabled (backend=%s mode=%s fullgraph=%s cudagraphs=off cache=%s)",
                    self.compile_backend,
                    self.compile_mode,
                    self.compile_fullgraph,
                    str(self.compile_cache_dir),
                )
            except Exception as exc:  # pragma: no cover
                self.use_compile = False
                logger.warning("torch.compile failed, fallback to eager. reason=%s", exc)

        logger.info("Model params: %s", f"{count_parameters(self.model):,}")
        logger.info(
            "Infra | channels_last=%s tf32=%s grad_ckpt=%s compile=%s mode=%s",
            self.channels_last,
            self.allow_tf32,
            bool(train_cfg.get("use_gradient_checkpointing", False)),
            self.use_compile,
            self.compile_mode if self.use_compile else "off",
        )

        self.use_amp = bool(train_cfg.get("use_amp", True) and device.type == "cuda")
        amp_dtype_cfg = str(train_cfg.get("amp_dtype", "bf16")).lower()
        if amp_dtype_cfg in {"fp16", "float16", "half"}:
            self.amp_dtype = torch.float16
        else:
            self.amp_dtype = torch.bfloat16
        if device.type != "cuda":
            self.use_amp = False
        self.use_grad_scaler = bool(self.use_amp and self.amp_dtype == torch.float16)
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.use_grad_scaler)

        use_fused_adamw = bool(train_cfg.get("fused_adamw", device.type == "cuda"))
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
            use_fused_adamw = False
        logger.info(
            "Precision | amp=%s amp_dtype=%s grad_scaler=%s fused_adamw=%s",
            self.use_amp,
            "fp16" if self.amp_dtype == torch.float16 else "bf16",
            self.use_grad_scaler,
            use_fused_adamw,
        )

        self.scheduler = None
        scheduler_name = str(train_cfg.get("scheduler", "cosine")).lower()
        if scheduler_name == "cosine":
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=max(1, int(train_cfg.get("num_epochs", 100))),
                eta_min=float(train_cfg.get("min_learning_rate", 1e-5)),
            )

        self.loss_fn = AdaCUTObjective(config)

        self.grad_clip_norm = float(train_cfg.get("grad_clip_norm", 1.0))
        self.accumulation_steps = max(1, int(train_cfg.get("accumulation_steps", 1)))
        self.log_interval = max(0, int(train_cfg.get("log_interval", 20)))
        self.use_tqdm = bool(train_cfg.get("use_tqdm", True))
        self.num_epochs = int(train_cfg.get("num_epochs", 100))
        self.save_interval = max(1, int(train_cfg.get("save_interval", 10)))
        self.full_eval_interval = max(0, int(train_cfg.get("full_eval_interval", 50)))
        self.run_full_eval_on_last_epoch = bool(train_cfg.get("full_eval_on_last_epoch", True))
        self.snapshot_source = bool(train_cfg.get("snapshot_source", False))

        if self.snapshot_source:
            self._snapshot_source()
        self.log_file = self.log_dir / f"training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        with open(self.log_file, "w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "epoch",
                    "loss",
                    "distill",
                    "code",
                    "code_pred_norm",
                    "code_ref_norm",
                    "cycle",
                    "struct",
                    "edge",
                    "stroke_gram",
                    "color_moment",
                    "push",
                    "delta_tv",
                    "style_spatial_tv",
                    "nce",
                    "semigroup",
                    "semigroup_samples",
                    "train_num_steps",
                    "train_step_size",
                    "train_style_strength",
                    "heavy_loss_slot_id",
                    "path_teacher_active",
                    "path_cycle_active",
                    "path_stroke_active",
                    "path_nce_active",
                    "path_semigroup_active",
                    "w_cycle_eff",
                    "w_struct_eff",
                    "w_edge_eff",
                    "w_nce_eff",
                    "w_semigroup_eff",
                    "transfer_ratio",
                    "lr",
                    "data_time_sec",
                    "compute_time_sec",
                    "epoch_time_sec",
                ]
            )

        self.global_step = 0
        self.start_epoch = 1
        self._maybe_resume(str(train_cfg.get("resume_checkpoint", "")))

    def _snapshot_source(self) -> None:
        """
        Copy current src into the run directory for traceability.
        """
        try:
            src_root = Path(__file__).resolve().parent
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            dst = self.checkpoint_dir / f"src_snapshot_{ts}"
            if dst.exists():
                return
            shutil.copytree(
                src_root,
                dst,
                ignore=shutil.ignore_patterns("__pycache__", "*.pyc", ".cache", "logs", "full_eval"),
            )
            logger.info("Saved src snapshot: %s", dst)
        except Exception as exc:  # pragma: no cover
            logger.warning("Failed to snapshot src: %s", exc)

    def _find_latest_checkpoint(self) -> Optional[Path]:
        ckpts = sorted(self.checkpoint_dir.glob("epoch_*.pt"))
        if not ckpts:
            return None
        return ckpts[-1]

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
        try:
            self.model.load_state_dict(model_state, strict=True)
        except RuntimeError as exc:
            incompatible = self.model.load_state_dict(model_state, strict=False)
            missing = list(getattr(incompatible, "missing_keys", []))
            unexpected = list(getattr(incompatible, "unexpected_keys", []))
            logger.warning(
                "Checkpoint loaded with non-strict mode due to key mismatch: %s | missing=%d unexpected=%d",
                exc,
                len(missing),
                len(unexpected),
            )
            if missing:
                logger.warning("Missing keys (first 12): %s", missing[:12])
            if unexpected:
                logger.warning("Unexpected keys (first 12): %s", unexpected[:12])

        if "optimizer_state_dict" in state:
            try:
                self.optimizer.load_state_dict(state["optimizer_state_dict"])
            except ValueError as exc:
                logger.warning("Skip optimizer state restore due to mismatch: %s", exc)
        if self.scheduler is not None and "scheduler_state_dict" in state and state["scheduler_state_dict"] is not None:
            try:
                self.scheduler.load_state_dict(state["scheduler_state_dict"])
            except ValueError as exc:
                logger.warning("Skip scheduler state restore due to mismatch: %s", exc)
        if "scaler_state_dict" in state:
            try:
                self.scaler.load_state_dict(state["scaler_state_dict"])
            except ValueError as exc:
                logger.warning("Skip scaler state restore due to mismatch: %s", exc)

        self.global_step = int(state.get("global_step", 0))
        self.start_epoch = int(state.get("epoch", 0)) + 1
        logger.info("Resumed from %s at epoch=%d global_step=%d", ckpt_path, self.start_epoch, self.global_step)

    def _move_batch(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        out = {}
        for k, v in batch.items():
            if torch.is_tensor(v):
                if self.channels_last and v.is_floating_point() and v.ndim == 4:
                    out[k] = v.to(self.device, non_blocking=True, memory_format=torch.channels_last)
                else:
                    out[k] = v.to(self.device, non_blocking=True)
            else:
                out[k] = v
        return out

    @staticmethod
    def _is_oom_error(exc: RuntimeError) -> bool:
        msg = str(exc).lower()
        return "out of memory" in msg or "cuda oom" in msg

    def _maybe_log_vram(self, epoch: int, step_idx: int) -> None:
        if self.device.type != "cuda" or self.log_vram_interval <= 0:
            return
        if step_idx % self.log_vram_interval != 0:
            return
        alloc_mb = torch.cuda.memory_allocated() / (1024**2)
        reserved_mb = torch.cuda.memory_reserved() / (1024**2)
        max_alloc_mb = torch.cuda.max_memory_allocated() / (1024**2)
        logger.info(
            "VRAM epoch=%d step=%d | alloc=%.1fMB reserved=%.1fMB peak=%.1fMB",
            epoch,
            step_idx,
            alloc_mb,
            reserved_mb,
            max_alloc_mb,
        )

    def step_scheduler(self) -> None:
        if self.scheduler is not None:
            self.scheduler.step()

    def train_epoch(self, dataloader: DataLoader, epoch: int) -> Dict[str, float]:
        self.model.train()
        epoch_start = time.time()
        self.loss_fn.set_progress(epoch=epoch, total_epochs=self.num_epochs)
        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()

        # Keep running sums as detached tensors to avoid per-step .item() sync stalls.
        metric_accum: Dict[str, torch.Tensor] = {}
        num_batches = 0
        data_time_total = 0.0
        compute_time_total = 0.0

        total_steps = len(dataloader)
        progress = tqdm(
            dataloader,
            total=total_steps,
            desc=f"Epoch {epoch}/{self.num_epochs}",
            leave=True,
            dynamic_ncols=True,
            mininterval=2.0,  # Reduce terminal I/O overhead in WSL
            disable=not self.use_tqdm,
        )

        self.optimizer.zero_grad(set_to_none=True)
        data_wait_start = time.perf_counter()
        for step_idx, raw_batch in enumerate(progress, start=1):
            step_enter = time.perf_counter()
            data_time_total += max(0.0, step_enter - data_wait_start)
            compute_start = time.perf_counter()
            loss_dict = None
            try:
                batch = self._move_batch(raw_batch)
                content = batch["content"]
                target_style = batch["target_style"]
                target_style_id = batch["target_style_id"]
                content_style_id = batch["content_style_id"]

                with torch.amp.autocast("cuda", enabled=self.use_amp, dtype=self.amp_dtype):
                    loss_dict = self.loss_fn.compute(
                        self.model,
                        content=content,
                        target_style=target_style,
                        target_style_id=target_style_id,
                        content_style_id=content_style_id,
                    )
                    loss = loss_dict["loss"]

                loss_to_backward = loss / self.accumulation_steps
                if self.use_grad_scaler:
                    self.scaler.scale(loss_to_backward).backward()
                else:
                    loss_to_backward.backward()

                should_step = (step_idx % self.accumulation_steps == 0)
                if should_step:
                    if self.grad_clip_norm > 0:
                        if self.use_grad_scaler:
                            self.scaler.unscale_(self.optimizer)
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip_norm)
                    if self.use_grad_scaler:
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                    else:
                        self.optimizer.step()
                    self.optimizer.zero_grad(set_to_none=True)
                    self.global_step += 1
                    if self.empty_cache_interval > 0 and self.device.type == "cuda":
                        if self.global_step % self.empty_cache_interval == 0:
                            torch.cuda.empty_cache()
                
                # Accumulate detached metrics without per-step host sync.
                for k, v in loss_dict.items():
                    if v is None:
                        continue
                    vd = v.detach()
                    if k in metric_accum:
                        metric_accum[k] = metric_accum[k] + vd
                    else:
                        metric_accum[k] = vd
                num_batches += 1

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
                        sgram=f"{_get_avg('stroke_gram'):.4f}",
                        cyc=f"{_get_avg('cycle'):.4f}",
                        semi=f"{_get_avg('semigroup'):.4f}",
                        semib=f"{_get_avg('semigroup_samples'):.1f}",
                        steps=f"{_get_avg('train_num_steps'):.1f}",
                        h=f"{_get_avg('train_step_size'):.2f}",
                        s=f"{_get_avg('train_style_strength'):.2f}",
                        pcy=f"{_get_avg('path_cycle_active'):.2f}",
                        pnce=f"{_get_avg('path_nce_active'):.2f}",
                        psemi=f"{_get_avg('path_semigroup_active'):.2f}",
                        wnce=f"{_get_avg('w_nce_eff'):.2f}",
                        xfer=f"{_get_avg('transfer_ratio'):.2f}",
                        data_ms=f"{(1000.0 * data_time_total / max(step_idx, 1)):.1f}",
                        comp_ms=f"{(1000.0 * compute_time_total / max(step_idx, 1)):.1f}",
                        it_s=f"{step_per_sec:.2f}",
                        eta=f"{eta:.1f}s",
                    )
                    if not self.use_tqdm:
                        logger.info(
                            "epoch %d step %d/%d | loss=%.4f sgram=%.4f cmoment=%.4f code=%.4f cpn=%.3f crn=%.3f cycle=%.4f semi=%.4f nce=%.4f steps=%.1f h=%.2f s=%.2f slot=%.0f pcy=%.2f pnce=%.2f psemi=%.2f | data %.1fms comp %.1fms | %.2f it/s eta %.1fs",
                            epoch,
                            step_idx,
                            total_steps,
                            _get_avg('loss'),
                            _get_avg('stroke_gram'),
                            _get_avg('color_moment'),
                            _get_avg('code'),
                            _get_avg('code_pred_norm'),
                            _get_avg('code_ref_norm'),
                            _get_avg('cycle'),
                            _get_avg('semigroup'),
                            _get_avg('nce'),
                            _get_avg('train_num_steps'),
                            _get_avg('train_step_size'),
                            _get_avg('train_style_strength'),
                            _get_avg('heavy_loss_slot_id'),
                            _get_avg('path_cycle_active'),
                            _get_avg('path_nce_active'),
                            _get_avg('path_semigroup_active'),
                            (1000.0 * data_time_total / max(step_idx, 1)),
                            (1000.0 * compute_time_total / max(step_idx, 1)),
                            step_per_sec,
                            eta,
                        )
                self._maybe_log_vram(epoch, step_idx)
            except RuntimeError as exc:
                if self.device.type == "cuda" and self._is_oom_error(exc) and self.skip_oom_batches:
                    logger.warning(
                        "CUDA OOM at epoch=%d step=%d, skipping batch (reduce batch_size if frequent).",
                        epoch,
                        step_idx,
                    )
                    self.optimizer.zero_grad(set_to_none=True)
                    torch.cuda.empty_cache()
                    continue
                raise
            finally:
                compute_time_total += max(0.0, time.perf_counter() - compute_start)
                data_wait_start = time.perf_counter()
                del loss_dict

        progress.close()

        # Flush leftover gradients when last batch is not divisible by accumulation_steps.
        if num_batches > 0 and (num_batches % self.accumulation_steps != 0):
            if self.grad_clip_norm > 0:
                if self.use_grad_scaler:
                    self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip_norm)
            if self.use_grad_scaler:
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                self.optimizer.step()
            self.optimizer.zero_grad(set_to_none=True)
            self.global_step += 1

        epoch_time = time.time() - epoch_start
        lr = float(self.optimizer.param_groups[0]["lr"])
        
        # Finalize metrics
        metrics: Dict[str, float] = {}
        denom = max(num_batches, 1)
        for k, v in metric_accum.items():
            metrics[k] = float((v / denom).item())
            
        # Ensure standard keys exist for CSV logging
        metrics["lr"] = lr
        metrics["data_time_sec"] = data_time_total
        metrics["compute_time_sec"] = compute_time_total
        metrics["epoch_time_sec"] = epoch_time
        
        # Fill missing keys with 0.0 for safety
        expected_keys = [
            "loss", "distill", "code", "code_pred_norm", "code_ref_norm",
            "cycle", "struct", "edge", "stroke_gram", "color_moment",
            "push", "delta_tv", "style_spatial_tv", "nce", "semigroup", "semigroup_samples",
            "train_num_steps", "train_step_size", "train_style_strength", "heavy_loss_slot_id",
            "path_teacher_active", "path_cycle_active", "path_stroke_active", "path_nce_active", "path_semigroup_active",
            "w_cycle_eff", "w_struct_eff",
            "w_edge_eff", "w_nce_eff", "w_semigroup_eff", "transfer_ratio",
            "data_time_sec", "compute_time_sec",
        ]
        for k in expected_keys:
            metrics.setdefault(k, 0.0)

        metrics.update({
            "data_time_sec": data_time_total,
            "compute_time_sec": compute_time_total,
            "epoch_time_sec": epoch_time,
        })

        if self.use_tqdm:
            tqdm.write(
                f"[Epoch {epoch}/{self.num_epochs}] "
                f"loss={metrics['loss']:.4f} "
                f"distill={metrics['distill']:.4f} "
                f"code={metrics['code']:.4f} "
                f"cpn={metrics['code_pred_norm']:.3f} crn={metrics['code_ref_norm']:.3f} "
                f"cycle={metrics['cycle']:.4f} "
                f"sgram={metrics['stroke_gram']:.4f} cmoment={metrics['color_moment']:.4f} "
                f"push={metrics['push']:.4f} dtv={metrics['delta_tv']:.4f} stv={metrics['style_spatial_tv']:.4f} "
                f"nce={metrics['nce']:.4f} semi={metrics['semigroup']:.4f} semib={metrics['semigroup_samples']:.1f} "
                f"steps={metrics['train_num_steps']:.1f} h={metrics['train_step_size']:.2f} s={metrics['train_style_strength']:.2f} slot={metrics['heavy_loss_slot_id']:.0f} "
                f"pth={metrics['path_teacher_active']:.2f} pcy={metrics['path_cycle_active']:.2f} pst={metrics['path_stroke_active']:.2f} pnce={metrics['path_nce_active']:.2f} psemi={metrics['path_semigroup_active']:.2f} "
                f"wcyc={metrics['w_cycle_eff']:.2f} wnce={metrics['w_nce_eff']:.2f} wsemi={metrics['w_semigroup_eff']:.2f} "
                f"xfer={metrics['transfer_ratio']:.2f} | data={data_time_total:.1f}s compute={compute_time_total:.1f}s total={epoch_time:.1f}s"
            )
        return metrics

    def log_epoch(self, epoch: int, metrics: Dict[str, float]) -> None:
        with open(self.log_file, "a", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    int(epoch),
                    float(metrics.get("loss", 0.0)),
                    float(metrics.get("distill", 0.0)),
                    float(metrics.get("code", 0.0)),
                    float(metrics.get("code_pred_norm", 0.0)),
                    float(metrics.get("code_ref_norm", 0.0)),
                    float(metrics.get("cycle", 0.0)),
                    float(metrics.get("struct", 0.0)),
                    float(metrics.get("edge", 0.0)),
                    float(metrics.get("stroke_gram", 0.0)),
                    float(metrics.get("color_moment", 0.0)),
                    float(metrics.get("push", 0.0)),
                    float(metrics.get("delta_tv", 0.0)),
                    float(metrics.get("style_spatial_tv", 0.0)),
                    float(metrics.get("nce", 0.0)),
                    float(metrics.get("semigroup", 0.0)),
                    float(metrics.get("semigroup_samples", 0.0)),
                    float(metrics.get("train_num_steps", 0.0)),
                    float(metrics.get("train_step_size", 0.0)),
                    float(metrics.get("train_style_strength", 0.0)),
                    float(metrics.get("heavy_loss_slot_id", 0.0)),
                    float(metrics.get("path_teacher_active", 0.0)),
                    float(metrics.get("path_cycle_active", 0.0)),
                    float(metrics.get("path_stroke_active", 0.0)),
                    float(metrics.get("path_nce_active", 0.0)),
                    float(metrics.get("path_semigroup_active", 0.0)),
                    float(metrics.get("w_cycle_eff", 0.0)),
                    float(metrics.get("w_struct_eff", 0.0)),
                    float(metrics.get("w_edge_eff", 0.0)),
                    float(metrics.get("w_nce_eff", 0.0)),
                    float(metrics.get("w_semigroup_eff", 0.0)),
                    float(metrics.get("transfer_ratio", 0.0)),
                    float(metrics.get("lr", 0.0)),
                    float(metrics.get("data_time_sec", 0.0)),
                    float(metrics.get("compute_time_sec", 0.0)),
                    float(metrics.get("epoch_time_sec", 0.0)),
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
            "scaler_state_dict": self.scaler.state_dict(),
            "config": self.config,
            "metrics": metrics,
        }
        torch.save(payload, path)
        logger.info("Saved checkpoint: %s", path)
        return path

    def run_full_evaluation(self, epoch: int, checkpoint_path: Optional[Path] = None) -> bool:
        """
        Launch external full evaluation script.
        """
        if checkpoint_path is None:
            checkpoint_path = self.checkpoint_dir / f"epoch_{epoch:04d}.pt"
        if not checkpoint_path.exists():
            logger.warning("Skip full eval: checkpoint missing: %s", checkpoint_path)
            return False

        utils_script = Path(__file__).resolve().parent / "utils" / "run_evaluation.py"
        if not utils_script.exists():
            logger.warning("Skip full eval: script not found: %s", utils_script)
            return False

        out_dir = self.full_eval_root / f"epoch_{epoch:04d}"
        out_dir.mkdir(parents=True, exist_ok=True)

        cfg_train = self.config.get("training", {})
        cfg_infer = self.config.get("inference", {})
        cfg_loss = self.config.get("loss", {})

        cmd = [
            sys.executable,
            str(utils_script),
            "--checkpoint",
            str(checkpoint_path),
            "--output",
            str(out_dir),
            "--num_steps",
            str(int(cfg_train.get("full_eval_num_steps", cfg_infer.get("num_steps", 1)))),
            "--step_size",
            str(float(cfg_train.get("full_eval_step_size", cfg_infer.get("step_size", 1.0)))),
            "--batch_size",
            str(int(cfg_train.get("full_eval_batch_size", 8))),
            "--max_src_samples",
            str(int(cfg_train.get("full_eval_max_src_samples", 30))),
            "--max_ref_compare",
            str(int(cfg_train.get("full_eval_max_ref_compare", 50))),
            "--max_ref_cache",
            str(int(cfg_train.get("full_eval_max_ref_cache", 256))),
            "--ref_feature_batch_size",
            str(int(cfg_train.get("full_eval_ref_feature_batch_size", 64))),
            "--style_ref_mode",
            str(cfg_train.get("full_eval_style_ref_mode", "prototype")),
            "--style_ref_count",
            str(int(cfg_train.get("full_eval_style_ref_count", 8))),
            "--style_ref_seed",
            str(int(cfg_train.get("full_eval_style_ref_seed", 2026))),
        ]
        full_eval_style_strength = cfg_train.get("full_eval_style_strength", cfg_infer.get("style_strength"))
        if full_eval_style_strength is not None:
            cmd += ["--style_strength", str(float(full_eval_style_strength))]
        full_eval_step_schedule = cfg_train.get("full_eval_step_schedule", cfg_infer.get("step_schedule"))
        if isinstance(full_eval_step_schedule, (list, tuple)):
            schedule_arg = ",".join(str(float(v)) for v in full_eval_step_schedule)
            if schedule_arg:
                cmd += ["--step_schedule", schedule_arg]
        elif full_eval_step_schedule is not None and str(full_eval_step_schedule).lower() not in {"", "none"}:
            cmd += ["--step_schedule", str(full_eval_step_schedule)]

        test_dir = cfg_train.get("test_image_dir", "")
        if test_dir:
            cmd += ["--test_dir", str(test_dir)]
        cache_dir = cfg_train.get("full_eval_cache_dir", "")
        if cache_dir:
            cmd += ["--cache_dir", str(cache_dir)]
        classifier_path = cfg_loss.get("style_classifier_ckpt", "")
        if classifier_path:
            cmd += ["--classifier_path", str(classifier_path)]
        if bool(cfg_train.get("full_eval_classifier_only", False)):
            cmd += ["--eval_classifier_only"]
        if bool(cfg_train.get("full_eval_disable_lpips", False)):
            cmd += ["--eval_disable_lpips"]

        log_path = self.log_dir / f"full_eval_epoch_{epoch:04d}.log"
        logger.info("Running full eval for epoch %d -> %s", epoch, out_dir)
        with open(log_path, "w", encoding="utf-8") as logf:
            proc = subprocess.run(cmd, cwd=str(Path(__file__).resolve().parent), stdout=logf, stderr=subprocess.STDOUT)
        if proc.returncode != 0:
            logger.error("Full eval failed for epoch %d (code=%d). See %s", epoch, proc.returncode, log_path)
            return False
        self._write_full_eval_history()
        logger.info("Full eval completed for epoch %d. Log: %s", epoch, log_path)
        return True

    def _write_full_eval_history(self) -> None:
        """
        Aggregate multi-round full_eval summaries into one history report.
        """
        rounds = []
        for epoch_dir in sorted(self.full_eval_root.glob("epoch_*")):
            summary_path = epoch_dir / "summary.json"
            if not summary_path.exists():
                continue
            try:
                with open(summary_path, "r", encoding="utf-8") as f:
                    summary = json.load(f)
            except Exception:
                continue
            name = epoch_dir.name
            try:
                epoch = int(name.split("_")[-1])
            except Exception:
                continue

            analysis = summary.get("analysis", {})
            transfer = analysis.get("style_transfer_ability", {})
            p2a = analysis.get("photo_to_art_performance", {})
            rounds.append(
                {
                    "epoch": epoch,
                    "summary_path": str(summary_path),
                    "transfer_clip_style": float(transfer.get("clip_style", 0.0)),
                    "transfer_content_lpips": float(transfer.get("content_lpips", 0.0)),
                    "transfer_classifier_acc": float(transfer.get("classifier_acc", 0.0)),
                    "photo_to_art_clip_style": float(p2a.get("clip_style", 0.0)),
                    "photo_to_art_classifier_acc": float(p2a.get("classifier_acc", 0.0)),
                }
            )

        if not rounds:
            return

        rounds.sort(key=lambda x: x["epoch"])
        latest = rounds[-1]
        mean = {
            "transfer_clip_style": sum(x["transfer_clip_style"] for x in rounds) / len(rounds),
            "transfer_content_lpips": sum(x["transfer_content_lpips"] for x in rounds) / len(rounds),
            "transfer_classifier_acc": sum(x["transfer_classifier_acc"] for x in rounds) / len(rounds),
            "photo_to_art_clip_style": sum(x["photo_to_art_clip_style"] for x in rounds) / len(rounds),
            "photo_to_art_classifier_acc": sum(x["photo_to_art_classifier_acc"] for x in rounds) / len(rounds),
        }
        best = {
            "best_transfer_classifier_acc": max(rounds, key=lambda x: x["transfer_classifier_acc"]),
            "best_transfer_clip_style": max(rounds, key=lambda x: x["transfer_clip_style"]),
            "best_photo_to_art_classifier_acc": max(rounds, key=lambda x: x["photo_to_art_classifier_acc"]),
            "best_photo_to_art_clip_style": max(rounds, key=lambda x: x["photo_to_art_clip_style"]),
            "best_transfer_content_lpips": min(rounds, key=lambda x: x["transfer_content_lpips"]),
        }

        payload = {
            "num_rounds": len(rounds),
            "latest": latest,
            "mean": mean,
            "best": best,
            "rounds": rounds,
            "updated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }
        history_path = self.full_eval_root / "summary_history.json"
        with open(history_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
