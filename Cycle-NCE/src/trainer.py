from __future__ import annotations

import csv
import gc
import json
import shutil
import logging
import os
import re
import subprocess
import sys
import time
from contextlib import contextmanager, nullcontext
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

import torch
from torch.profiler import ProfilerActivity, profile, schedule, tensorboard_trace_handler
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


def _is_rtx_30_series(name: str) -> bool:
    n = str(name).lower()
    if "rtx 30" in n:
        return True
    return bool(re.search(r"\b30\d{2}\b", n))


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
        self.skip_oom_batches = bool(train_cfg.get("skip_oom_batches", True))
        self.gpu_name = ""
        self.gpu_capability = ""
        self.gpu_total_vram_gb = 0.0
        self.is_rtx_30_series = False
        if device.type == "cuda" and torch.cuda.is_available():
            try:
                dev_idx = torch.cuda.current_device()
                props = torch.cuda.get_device_properties(dev_idx)
                self.gpu_name = str(getattr(props, "name", ""))
                self.gpu_capability = f"{int(getattr(props, 'major', 0))}.{int(getattr(props, 'minor', 0))}"
                self.gpu_total_vram_gb = float(getattr(props, "total_memory", 0)) / float(1024**3)
                self.is_rtx_30_series = _is_rtx_30_series(self.gpu_name)
            except Exception:  # pragma: no cover
                pass

        torch.set_float32_matmul_precision("high")
        self.allow_tf32 = bool(train_cfg.get("allow_tf32", True))
        if torch.cuda.is_available():
            torch.backends.cuda.matmul.allow_tf32 = self.allow_tf32
            torch.backends.cudnn.allow_tf32 = self.allow_tf32
            torch.backends.cudnn.benchmark = True

        requested_channels_last = bool(train_cfg.get("channels_last", False) and device.type == "cuda")
        self.channels_last = bool(requested_channels_last)
        self.empty_cache_interval = max(0, int(train_cfg.get("empty_cache_interval", 0)))
        self.log_vram_interval = max(0, int(train_cfg.get("log_vram_interval", 0)))
        self.gc_collect_interval = max(0, int(train_cfg.get("gc_collect_interval", 0)))
        self.trace_vram_steps = max(0, int(train_cfg.get("trace_vram_steps", 1)))
        requested_loss_timing_interval = max(0, int(train_cfg.get("loss_timing_interval", 0)))
        # Force-disable CUDA event timing in training path to avoid synchronize overhead.
        self.loss_timing_interval = 0
        self.strict_batch_sanity = bool(train_cfg.get("strict_batch_sanity", True))
        self.strict_batch_sanity_interval = max(1, int(train_cfg.get("strict_batch_sanity_interval", 1)))
        requested_cuda_sync_debug = bool(train_cfg.get("cuda_sync_debug", False))
        # Force-disable explicit synchronize path for throughput.
        self.cuda_sync_debug = False
        if requested_cuda_sync_debug:
            logger.warning("cuda_sync_debug requested but forcibly disabled for throughput stability.")
        if requested_loss_timing_interval > 0:
            logger.warning("loss_timing_interval=%d requested but forcibly disabled for throughput.", requested_loss_timing_interval)
        self.enable_profiler = bool(train_cfg.get("enable_profiler", False))
        self.nsight_cuda_profile = bool(train_cfg.get("nsight_cuda_profile", False))
        self.nsight_capture_start_step = max(0, int(train_cfg.get("nsight_capture_start_step", 10)))
        self.nsight_capture_steps = max(1, int(train_cfg.get("nsight_capture_steps", 50)))
        self.nsight_capture_end_step = self.nsight_capture_start_step + self.nsight_capture_steps
        self.nsight_nvtx = bool(train_cfg.get("nsight_nvtx", False))
        self._cuda_profiler_started = False
        self._cuda_profiler_stopped = False
        if self.nsight_cuda_profile and device.type != "cuda":
            logger.warning("nsight_cuda_profile requested but CUDA is unavailable, disabling.")
            self.nsight_cuda_profile = False
        if self.enable_profiler and self.nsight_cuda_profile:
            logger.warning("Both torch profiler and nsight_cuda_profile are enabled; this can distort profiling timelines.")
        self.profiler_dir = ""
        self.profiler_wait = max(0, int(train_cfg.get("profiler_wait", 1)))
        self.profiler_warmup = max(0, int(train_cfg.get("profiler_warmup", 1)))
        self.profiler_active = max(1, int(train_cfg.get("profiler_active", 3)))
        self.profiler_repeat = max(1, int(train_cfg.get("profiler_repeat", 1)))

        model_cfg = config.get("model", {})
        grad_ckpt_cfg = bool(train_cfg.get("use_gradient_checkpointing", False))
        if grad_ckpt_cfg:
            logger.info("Gradient checkpointing enabled (trades compute for lower VRAM).")
        self.model = build_model_from_config(
            model_cfg,
            use_checkpointing=grad_ckpt_cfg,
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
        self.profiler_dir = str(train_cfg.get("profiler_dir", str((self.log_dir / "profiler").resolve())))
        self.oom_dump_dir = self.log_dir / "oom_reports"
        self.oom_dump_dir.mkdir(parents=True, exist_ok=True)
        self.full_eval_root = self.checkpoint_dir / "full_eval"
        self.full_eval_root.mkdir(parents=True, exist_ok=True)

        requested_use_compile = bool(train_cfg.get("use_compile", False))
        disable_compile_on_rtx30 = bool(train_cfg.get("disable_compile_on_rtx30", True))
        if requested_use_compile and self.is_rtx_30_series and disable_compile_on_rtx30:
            logger.warning(
                "use_compile=True requested but disabled for RTX 30 series (%s) to prioritize long-run stability.",
                self.gpu_name or "unknown",
            )
        self.use_compile = bool(
            requested_use_compile
            and device.type == "cuda"
            and not (self.is_rtx_30_series and disable_compile_on_rtx30)
        )
        self.compile_backend = str(train_cfg.get("compile_backend", _COMPILE_BACKEND))
        self.compile_mode = str(train_cfg.get("compile_mode", _COMPILE_MODE))
        self.compile_fullgraph = bool(train_cfg.get("compile_fullgraph", _COMPILE_FULLGRAPH))
        self.compile_disable_cudagraphs = bool(train_cfg.get("compile_disable_cudagraphs", True))
        self.compile_cache_dir = (self.checkpoint_dir / "torch_compile_cache").resolve()
        if requested_use_compile and device.type != "cuda":
            logger.warning("use_compile=True requested but CUDA is unavailable; fallback to eager.")
        if self.use_compile:
            try:
                (self.compile_cache_dir / "inductor").mkdir(parents=True, exist_ok=True)
                (self.compile_cache_dir / "triton").mkdir(parents=True, exist_ok=True)
                os.environ["TORCHINDUCTOR_CACHE_DIR"] = str((self.compile_cache_dir / "inductor"))
                os.environ["TRITON_CACHE_DIR"] = str((self.compile_cache_dir / "triton"))
                if self.compile_disable_cudagraphs:
                    os.environ["TORCHINDUCTOR_CUDAGRAPHS"] = "0"
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
                    "torch.compile enabled (backend=%s mode=%s fullgraph=%s cudagraphs=%s cache=%s)",
                    self.compile_backend,
                    self.compile_mode,
                    self.compile_fullgraph,
                    "off" if self.compile_disable_cudagraphs else "on",
                    str(self.compile_cache_dir),
                )
            except Exception as exc:  # pragma: no cover
                self.use_compile = False
                logger.warning("torch.compile failed, fallback to eager. reason=%s", exc)

        logger.info("Model params: %s", f"{count_parameters(self.model):,}")
        logger.info(
            "Infra | channels_last=%s tf32=%s grad_ckpt=%s compile=%s backend=%s mode=%s fullgraph=%s cudagraphs=%s gc_collect_interval=%d loss_timing_interval=%d strict_batch_sanity=%s strict_batch_sanity_interval=%d cuda_sync_debug=%s profiler=%s alloc_conf=%s",
            self.channels_last,
            self.allow_tf32,
            grad_ckpt_cfg,
            self.use_compile,
            self.compile_backend if self.use_compile else "off",
            self.compile_mode if self.use_compile else "off",
            self.compile_fullgraph if self.use_compile else False,
            ("off" if getattr(self, "compile_disable_cudagraphs", True) else "on") if self.use_compile else "off",
            self.gc_collect_interval,
            self.loss_timing_interval,
            self.strict_batch_sanity,
            self.strict_batch_sanity_interval,
            self.cuda_sync_debug,
            self.enable_profiler,
            os.environ.get("PYTORCH_ALLOC_CONF", os.environ.get("PYTORCH_CUDA_ALLOC_CONF", "")),
        )
        logger.info(
            "Nsight | nvtx=%s cuda_profile=%s capture_start_step=%d capture_steps=%d",
            self.nsight_nvtx,
            self.nsight_cuda_profile,
            self.nsight_capture_start_step,
            self.nsight_capture_steps,
        )

        self.use_amp = bool(train_cfg.get("use_amp", False) and device.type == "cuda")
        amp_dtype_cfg = str(train_cfg.get("amp_dtype", "fp16")).lower()
        prefer_fp16_on_rtx30 = bool(train_cfg.get("prefer_fp16_on_rtx30", True))
        if (
            self.use_amp
            and self.is_rtx_30_series
            and prefer_fp16_on_rtx30
            and amp_dtype_cfg in {"bf16", "bfloat16"}
        ):
            logger.warning(
                "amp_dtype=%s requested on RTX 30 series (%s), forcing fp16 for better stability/perf balance.",
                amp_dtype_cfg,
                self.gpu_name or "unknown",
            )
            amp_dtype_cfg = "fp16"
        if amp_dtype_cfg in {"fp16", "float16", "half"}:
            self.amp_dtype = torch.float16
        else:
            self.amp_dtype = torch.bfloat16
        if device.type != "cuda":
            self.use_amp = False
        default_use_grad_scaler = bool(self.use_amp and self.amp_dtype == torch.float16)
        self.use_grad_scaler = bool(train_cfg.get("use_grad_scaler", default_use_grad_scaler))
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.use_grad_scaler)

        requested_fused_adamw = bool(train_cfg.get("fused_adamw", device.type == "cuda"))
        use_fused_adamw = bool(requested_fused_adamw and device.type == "cuda")
        if requested_fused_adamw and device.type != "cuda":
            logger.warning("fused_adamw=True requested but CUDA is unavailable; using regular AdamW.")
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
            if requested_fused_adamw:
                logger.warning("Fused AdamW is not supported in this torch build; fallback to regular AdamW.")
        logger.info(
            "Precision | amp=%s amp_dtype=%s grad_scaler=%s fused_adamw=%s prefer_fp16_on_rtx30=%s",
            self.use_amp,
            "fp16" if self.amp_dtype == torch.float16 else "bf16",
            self.use_grad_scaler,
            use_fused_adamw,
            prefer_fp16_on_rtx30,
        )
        logger.info(
            "GPU | name=%s capability=%s vram_gb=%.2f rtx30=%s",
            self.gpu_name or "unknown",
            self.gpu_capability or "unknown",
            self.gpu_total_vram_gb,
            self.is_rtx_30_series,
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
        logger.info(
            "SWD objective | feature_space=%s patch_sizes=%s projections=%d padding=%s max_patches=%d",
            getattr(self.loss_fn, "swd_feature_space", "unknown"),
            getattr(self.loss_fn, "swd_patch_sizes", []),
            int(getattr(self.loss_fn, "swd_num_projections", 0)),
            getattr(self.loss_fn, "swd_padding_mode", "same"),
            int(getattr(self.loss_fn, "swd_max_patches", 0)),
        )

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
                    "semigroup",
                    "swd",
                    "moment",
                    "identity",
                    "identity_ratio",
                    "delta_tv",
                    "delta_l2",
                    "output_tv",
                    "swd_valid_samples",
                    "shortcut_mu_delta",
                    "shortcut_std_delta",
                    "shortcut_texture_gain",
                    "shortcut_lsi",
                    "swd_weight",
                    "train_num_steps",
                    "train_step_size",
                    "train_style_strength",
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
                if v.device == self.device:
                    out[k] = v
                elif self.channels_last and v.is_floating_point() and v.ndim == 4:
                    out[k] = v.to(self.device, non_blocking=True, memory_format=torch.channels_last)
                else:
                    out[k] = v.to(self.device, non_blocking=True)
            else:
                out[k] = v
        return out

    def _validate_batch_sanity_cpu(self, batch: Dict[str, torch.Tensor]) -> None:
        content = batch.get("content")
        target_style = batch.get("target_style")
        target_style_id = batch.get("target_style_id")
        source_style_id = batch.get("source_style_id")
        if not torch.is_tensor(content) or not torch.is_tensor(target_style) or not torch.is_tensor(target_style_id):
            raise RuntimeError("Missing required batch keys: content/target_style/target_style_id")
        if source_style_id is not None and not torch.is_tensor(source_style_id):
            raise RuntimeError("source_style_id must be a tensor when provided.")
        # Keep this check on CPU tensors to avoid device sync stalls.
        if not torch.isfinite(content).all() or not torch.isfinite(target_style).all():
            raise RuntimeError("Non-finite values in batch tensors (content/target_style).")
        n_styles = int(getattr(self.model, "num_styles", 0))
        if n_styles > 0:
            sid_min = int(target_style_id.min().item())
            sid_max = int(target_style_id.max().item())
            if sid_min < 0 or sid_max >= n_styles:
                raise RuntimeError(
                    f"target_style_id out of range: min={sid_min} max={sid_max} valid=[0,{n_styles-1}]"
                )
            if source_style_id is not None:
                src_min = int(source_style_id.min().item())
                src_max = int(source_style_id.max().item())
                if src_min < 0 or src_max >= n_styles:
                    raise RuntimeError(
                        f"source_style_id out of range: min={src_min} max={src_max} valid=[0,{n_styles-1}]"
                    )

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

    def _cuda_memory_snapshot(self) -> Dict[str, float]:
        if self.device.type != "cuda":
            return {}
        stats = torch.cuda.memory_stats()
        free_bytes, total_bytes = torch.cuda.mem_get_info()
        snapshot = {
            "allocated_mb": float(torch.cuda.memory_allocated() / (1024**2)),
            "reserved_mb": float(torch.cuda.memory_reserved() / (1024**2)),
            "peak_allocated_mb": float(torch.cuda.max_memory_allocated() / (1024**2)),
            "peak_reserved_mb": float(torch.cuda.max_memory_reserved() / (1024**2)),
            "inactive_split_mb": float(stats.get("inactive_split_bytes.all.current", 0) / (1024**2)),
            "active_bytes_mb": float(stats.get("active_bytes.all.current", 0) / (1024**2)),
            "cuda_free_mb": float(free_bytes / (1024**2)),
            "cuda_total_mb": float(total_bytes / (1024**2)),
        }
        return snapshot

    def _trace_vram(self, *, epoch: int, step_idx: int, phase: str) -> None:
        if self.device.type != "cuda":
            return
        if self.trace_vram_steps <= 0 or step_idx > self.trace_vram_steps:
            return
        snap = self._cuda_memory_snapshot()
        logger.info(
            "VRAM_TRACE epoch=%d step=%d phase=%s alloc=%.1fMB reserved=%.1fMB peak=%.1fMB inactive_split=%.1fMB free=%.1fMB",
            epoch,
            step_idx,
            phase,
            snap.get("allocated_mb", 0.0),
            snap.get("reserved_mb", 0.0),
            snap.get("peak_allocated_mb", 0.0),
            snap.get("inactive_split_mb", 0.0),
            snap.get("cuda_free_mb", 0.0),
        )

    def _dump_oom_report(self, *, epoch: int, step_idx: int, exc: RuntimeError) -> None:
        if self.device.type != "cuda":
            return
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        base = self.oom_dump_dir / f"oom_e{epoch:04d}_s{step_idx:06d}_{ts}"
        snapshot = self._cuda_memory_snapshot()
        payload = {
            "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "epoch": int(epoch),
            "step": int(step_idx),
            "error": str(exc),
            "alloc_conf": os.environ.get("PYTORCH_ALLOC_CONF", os.environ.get("PYTORCH_CUDA_ALLOC_CONF", "")),
            "snapshot": snapshot,
        }
        try:
            with open(f"{base}.json", "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2, ensure_ascii=False)
            with open(f"{base}.txt", "w", encoding="utf-8") as f:
                f.write(torch.cuda.memory_summary(device=self.device, abbreviated=False))
            logger.error("OOM diagnostics saved: %s.[json|txt]", str(base))
        except Exception as dump_exc:  # pragma: no cover
            logger.error("Failed to save OOM diagnostics: %s", dump_exc)

    @contextmanager
    def _nvtx_range(self, name: str):
        if not (self.nsight_nvtx and self.device.type == "cuda"):
            yield
            return
        try:
            torch.cuda.nvtx.range_push(name)
            yield
        finally:
            try:
                torch.cuda.nvtx.range_pop()
            except Exception:  # pragma: no cover
                pass

    def _maybe_toggle_cuda_profiler(self) -> None:
        if not (self.nsight_cuda_profile and self.device.type == "cuda"):
            return
        try:
            cudart = torch.cuda.cudart()
        except Exception as exc:  # pragma: no cover
            logger.warning("Failed to access CUDA runtime for profiler control: %s", exc)
            self.nsight_cuda_profile = False
            return

        if (not self._cuda_profiler_started) and self.global_step >= self.nsight_capture_start_step:
            try:
                cudart.cudaProfilerStart()
                self._cuda_profiler_started = True
                logger.info(
                    "CUDA profiler started at global_step=%d (capture range: [%d, %d))",
                    self.global_step,
                    self.nsight_capture_start_step,
                    self.nsight_capture_end_step,
                )
            except Exception as exc:  # pragma: no cover
                logger.warning("cudaProfilerStart failed: %s", exc)
                self.nsight_cuda_profile = False
                return

        if (
            self._cuda_profiler_started
            and (not self._cuda_profiler_stopped)
            and self.global_step >= self.nsight_capture_end_step
        ):
            try:
                cudart.cudaProfilerStop()
                self._cuda_profiler_stopped = True
                logger.info("CUDA profiler stopped at global_step=%d", self.global_step)
            except Exception as exc:  # pragma: no cover
                logger.warning("cudaProfilerStop failed: %s", exc)
                self.nsight_cuda_profile = False

    def step_scheduler(self) -> None:
        if self.scheduler is not None:
            self.scheduler.step()

    def train_epoch(self, dataloader: DataLoader, epoch: int) -> Dict[str, float]:
        self.model.train()
        epoch_start = time.time()
        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()

        # Keep running sums as detached tensors to avoid per-step .item() sync stalls.
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
        profiler_enabled = bool(self.enable_profiler and self.device.type == "cuda")
        profiler_ctx = (
            profile(
                activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                schedule=schedule(
                    wait=self.profiler_wait,
                    warmup=self.profiler_warmup,
                    active=self.profiler_active,
                    repeat=self.profiler_repeat,
                ),
                on_trace_ready=tensorboard_trace_handler(self.profiler_dir),
                record_shapes=True,
                profile_memory=True,
                with_stack=False,
            )
            if profiler_enabled
            else nullcontext()
        )
        if profiler_enabled:
            logger.info("Profiler enabled -> %s", self.profiler_dir)

        with profiler_ctx as prof:
            for step_idx, raw_batch in enumerate(progress, start=1):
                self._maybe_toggle_cuda_profiler()
                step_enter = time.perf_counter()
                data_time_total += max(0.0, step_enter - data_wait_start)
                step_compute_start = time.perf_counter()
                transfer_step = 0.0
                fwd_loss_step = 0.0
                backward_step = 0.0
                optimizer_step = 0.0
                batch = None
                content = None
                target_style = None
                target_style_id = None
                source_style_id = None
                loss_dict = None
                loss = None
                try:
                    if self.strict_batch_sanity and (step_idx % self.strict_batch_sanity_interval == 0):
                        with self._nvtx_range("batch_sanity_cpu"):
                            self._validate_batch_sanity_cpu(raw_batch)
                    t0 = time.perf_counter()
                    with self._nvtx_range("move_batch"):
                        batch = self._move_batch(raw_batch)
                    transfer_step += max(0.0, time.perf_counter() - t0)
                    self._trace_vram(epoch=epoch, step_idx=step_idx, phase="after_move_batch")
                    content = batch["content"]
                    target_style = batch["target_style"]
                    target_style_id = batch["target_style_id"]
                    source_style_id = batch.get("source_style_id")
                    enable_loss_timing = bool(self.loss_timing_interval > 0 and (step_idx % self.loss_timing_interval == 0))

                    t0 = time.perf_counter()
                    with self._nvtx_range("loss_compute"):
                        with torch.amp.autocast("cuda", enabled=self.use_amp, dtype=self.amp_dtype):
                            loss_dict = self.loss_fn.compute(
                                self.model,
                                content=content,
                                target_style=target_style,
                                target_style_id=target_style_id,
                                source_style_id=source_style_id,
                                debug_timing=enable_loss_timing,
                            )
                            loss = loss_dict["loss"]
                    if loss is None or (not torch.isfinite(loss).all()):
                        logger.warning(
                            "Non-finite loss detected at epoch=%d step=%d; skipping batch for stability.",
                            epoch,
                            step_idx,
                        )
                        self.optimizer.zero_grad(set_to_none=True)
                        data_wait_start = time.perf_counter()
                        continue
                    fwd_loss_step += max(0.0, time.perf_counter() - t0)
                    if step_idx <= self.trace_vram_steps and "loss_vram_total_alloc_mb" in loss_dict:
                        def _vram_metric(name: str) -> float:
                            v = loss_dict.get(name)
                            if v is None:
                                return 0.0
                            return float(v.detach().item())
                        logger.info(
                            "LOSS_VRAM epoch=%d step=%d | pred=%.1fMB style=%+.1fMB delta=%+.1fMB semigroup=%+.1fMB total=%+.1fMB alloc_now=%.1fMB peak_from_start=%.1fMB",
                            epoch,
                            step_idx,
                            _vram_metric("loss_vram_pred_alloc_mb"),
                            _vram_metric("loss_vram_style_delta_mb"),
                            _vram_metric("loss_vram_delta_delta_mb"),
                            _vram_metric("loss_vram_semigroup_delta_mb"),
                            _vram_metric("loss_vram_total_delta_mb"),
                            _vram_metric("loss_vram_total_alloc_mb"),
                            _vram_metric("loss_vram_total_peak_from_start_mb"),
                        )
                    if "loss_time_total_ms" in (loss_dict or {}):
                        def _time_metric(name: str) -> float:
                            v = loss_dict.get(name)
                            if v is None:
                                return 0.0
                            return float(v.detach().item())
                        logger.info(
                            "LOSS_TIME epoch=%d step=%d | pred=%.2fms style=%.2fms delta=%.2fms semigroup=%.2fms total=%.2fms",
                            epoch,
                            step_idx,
                            _time_metric("loss_time_pred_ms"),
                            _time_metric("loss_time_style_ms"),
                            _time_metric("loss_time_delta_ms"),
                            _time_metric("loss_time_semigroup_ms"),
                            _time_metric("loss_time_total_ms"),
                        )
                    self._trace_vram(epoch=epoch, step_idx=step_idx, phase="after_loss_compute")

                    loss_to_backward = loss / self.accumulation_steps
                    t0 = time.perf_counter()
                    with self._nvtx_range("backward"):
                        if self.use_grad_scaler:
                            self.scaler.scale(loss_to_backward).backward()
                        else:
                            loss_to_backward.backward()
                    backward_step += max(0.0, time.perf_counter() - t0)
                    self._trace_vram(epoch=epoch, step_idx=step_idx, phase="after_backward")

                    should_step = (step_idx % self.accumulation_steps == 0)
                    if should_step:
                        t0 = time.perf_counter()
                        with self._nvtx_range("optimizer_step"):
                            grad_norm = None
                            if self.grad_clip_norm > 0:
                                if self.use_grad_scaler:
                                    self.scaler.unscale_(self.optimizer)
                                grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip_norm)
                            if grad_norm is not None and (not torch.isfinite(grad_norm)):
                                logger.warning(
                                    "Non-finite grad norm at epoch=%d step=%d; skip optimizer step and clear grads.",
                                    epoch,
                                    step_idx,
                                )
                                self.optimizer.zero_grad(set_to_none=True)
                                data_wait_start = time.perf_counter()
                                continue
                            if self.use_grad_scaler:
                                self.scaler.step(self.optimizer)
                                self.scaler.update()
                            else:
                                self.optimizer.step()
                            self.optimizer.zero_grad(set_to_none=True)
                        optimizer_step += max(0.0, time.perf_counter() - t0)
                        self._trace_vram(epoch=epoch, step_idx=step_idx, phase="after_optimizer_step")
                        self.global_step += 1
                        if self.gc_collect_interval > 0 and (self.global_step % self.gc_collect_interval == 0):
                            gc.collect()
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
                except RuntimeError as exc:
                    if self._is_oom_error(exc):
                        self._dump_oom_report(epoch=epoch, step_idx=step_idx, exc=exc)
                        if self.skip_oom_batches and self.device.type == "cuda":
                            logger.warning(
                                "OOM at epoch=%d step=%d. skip_oom_batches=True, dropping batch and continuing.",
                                epoch,
                                step_idx,
                            )
                            self.optimizer.zero_grad(set_to_none=True)
                            gc.collect()
                            torch.cuda.empty_cache()
                            data_wait_start = time.perf_counter()
                            continue
                    raise

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
                        semigroup=f"{_get_avg('semigroup'):.4f}",
                        swd=f"{_get_avg('swd'):.4f}",
                        moment=f"{_get_avg('moment'):.4f}",
                        idt=f"{_get_avg('identity'):.4f}",
                        idr=f"{_get_avg('identity_ratio'):.2f}",
                        lsi=f"{_get_avg('shortcut_lsi'):.3f}",
                        w_swd=f"{_get_avg('swd_weight'):.2f}",
                        sN=f"{_get_avg('swd_valid_samples'):.1f}",
                        steps=f"{_get_avg('train_num_steps'):.1f}",
                        h=f"{_get_avg('train_step_size'):.2f}",
                        s=f"{_get_avg('train_style_strength'):.2f}",
                        data_ms=f"{(1000.0 * data_time_total / max(step_idx, 1)):.1f}",
                        comp_ms=f"{(1000.0 * compute_time_total / max(step_idx, 1)):.1f}",
                        it_s=f"{step_per_sec:.2f}",
                        eta=f"{eta:.1f}s",
                    )
                    if not self.use_tqdm:
                        logger.info(
                            "epoch %d step %d/%d | loss=%.4f semigroup=%.4f swd=%.4f moment=%.4f idt=%.4f idr=%.2f dtv=%.4f dl2=%.4f otv=%.4f lsi=%.3f mu=%.4f std=%.4f tex=%.4f swdN=%.1f steps=%.1f h=%.2f s=%.2f | data %.1fms comp %.1fms | %.2f it/s eta %.1fs",
                            epoch,
                            step_idx,
                            total_steps,
                            _get_avg('loss'),
                            _get_avg('semigroup'),
                            _get_avg('swd'),
                            _get_avg('moment'),
                            _get_avg('identity'),
                            _get_avg('identity_ratio'),
                            _get_avg('delta_tv'),
                            _get_avg('delta_l2'),
                            _get_avg('output_tv'),
                            _get_avg('shortcut_lsi'),
                            _get_avg('shortcut_mu_delta'),
                            _get_avg('shortcut_std_delta'),
                            _get_avg('shortcut_texture_gain'),
                            _get_avg('swd_valid_samples'),
                            _get_avg('train_num_steps'),
                            _get_avg('train_step_size'),
                            _get_avg('train_style_strength'),
                            (1000.0 * data_time_total / max(step_idx, 1)),
                            (1000.0 * compute_time_total / max(step_idx, 1)),
                            step_per_sec,
                            eta,
                        )
                self._maybe_log_vram(epoch, step_idx)
                step_elapsed = max(0.0, time.perf_counter() - step_compute_start)
                step_compute = transfer_step + fwd_loss_step + backward_step + optimizer_step
                step_overhead = max(0.0, step_elapsed - step_compute)
                transfer_time_total += transfer_step
                fwd_loss_time_total += fwd_loss_step
                backward_time_total += backward_step
                optimizer_time_total += optimizer_step
                step_overhead_time_total += step_overhead
                compute_time_total += step_compute
                data_wait_start = time.perf_counter()
                if profiler_enabled and prof is not None:
                    prof.step()
                del loss
                del loss_dict
                del target_style_id
                del source_style_id
                del target_style
                del content
                del batch

        progress.close()

        # Flush leftover gradients when last batch is not divisible by accumulation_steps.
        if num_batches > 0 and (num_batches % self.accumulation_steps != 0):
            grad_norm = None
            if self.grad_clip_norm > 0:
                if self.use_grad_scaler:
                    self.scaler.unscale_(self.optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip_norm)
            if grad_norm is None or torch.isfinite(grad_norm):
                if self.use_grad_scaler:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()
            else:
                logger.warning("Non-finite grad norm at epoch tail; skipped optimizer step.")
            self.optimizer.zero_grad(set_to_none=True)
            self.global_step += 1
        self._maybe_toggle_cuda_profiler()

        epoch_time = time.time() - epoch_start
        lr = float(self.optimizer.param_groups[0]["lr"])
        batch_size = int(getattr(dataloader, "batch_size", 0) or 0)
        samples_seen = int(num_batches * batch_size) if batch_size > 0 else 0
        samples_per_sec = float(samples_seen / max(epoch_time, 1e-6)) if samples_seen > 0 else 0.0
        compute_samples_per_sec = float(samples_seen / max(compute_time_total, 1e-6)) if samples_seen > 0 else 0.0
        
        # Finalize metrics
        metrics: Dict[str, float] = {}
        denom = max(num_batches, 1)
        for k, v in metric_accum.items():
            metrics[k] = float((v / denom).item())
            
        # Ensure standard keys exist for CSV logging
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
        
        # Fill missing keys with 0.0 for safety
        expected_keys = [
            "loss", "semigroup", "swd", "moment", "identity", "identity_ratio",
            "delta_tv", "delta_l2", "output_tv",
            "swd_valid_samples", "shortcut_mu_delta", "shortcut_std_delta", "shortcut_texture_gain", "shortcut_lsi",
            "train_num_steps", "train_step_size", "train_style_strength",
            "data_time_sec", "transfer_time_sec", "fwd_loss_time_sec", "backward_time_sec",
            "optimizer_time_sec", "step_overhead_time_sec", "compute_time_sec",
            "samples_seen", "samples_per_sec", "compute_samples_per_sec",
        ]
        for k in expected_keys:
            metrics.setdefault(k, 0.0)

        metrics.update({
            "data_time_sec": data_time_total,
            "transfer_time_sec": transfer_time_total,
            "fwd_loss_time_sec": fwd_loss_time_total,
            "backward_time_sec": backward_time_total,
            "optimizer_time_sec": optimizer_time_total,
            "step_overhead_time_sec": step_overhead_time_total,
            "compute_time_sec": compute_time_total,
            "epoch_time_sec": epoch_time,
            "samples_seen": float(samples_seen),
            "samples_per_sec": samples_per_sec,
            "compute_samples_per_sec": compute_samples_per_sec,
        })

        if self.use_tqdm:
            tqdm.write(
                f"[Epoch {epoch}/{self.num_epochs}] "
                f"loss={metrics['loss']:.4f} "
                f"semigroup={metrics['semigroup']:.4f} "
                f"swd={metrics['swd']:.4f} moment={metrics['moment']:.4f} "
                f"idt={metrics['identity']:.4f} idr={metrics['identity_ratio']:.2f} "
                f"dtv={metrics['delta_tv']:.4f} dl2={metrics['delta_l2']:.4f} otv={metrics['output_tv']:.4f} "
                f"lsi={metrics['shortcut_lsi']:.3f} mu={metrics['shortcut_mu_delta']:.4f} std={metrics['shortcut_std_delta']:.4f} tex={metrics['shortcut_texture_gain']:.4f} swdN={metrics['swd_valid_samples']:.1f} "
                f"steps={metrics['train_num_steps']:.1f} h={metrics['train_step_size']:.2f} s={metrics['train_style_strength']:.2f} "
                f"| data={data_time_total:.1f}s transfer={transfer_time_total:.1f}s fwd={fwd_loss_time_total:.1f}s "
                f"bwd={backward_time_total:.1f}s opt={optimizer_time_total:.1f}s overhead={step_overhead_time_total:.1f}s "
                f"compute={compute_time_total:.1f}s total={epoch_time:.1f}s "
                f"| samples={samples_seen} sps={samples_per_sec:.1f} compute_sps={compute_samples_per_sec:.1f}"
            )
        return metrics

    def log_epoch(self, epoch: int, metrics: Dict[str, float]) -> None:
        with open(self.log_file, "a", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    int(epoch),
                    float(metrics.get("loss", 0.0)),
                    float(metrics.get("semigroup", 0.0)),
                    float(metrics.get("swd", 0.0)),
                    float(metrics.get("moment", 0.0)),
                    float(metrics.get("identity", 0.0)),
                    float(metrics.get("identity_ratio", 0.0)),
                    float(metrics.get("delta_tv", 0.0)),
                    float(metrics.get("delta_l2", 0.0)),
                    float(metrics.get("output_tv", 0.0)),
                    float(metrics.get("swd_valid_samples", 0.0)),
                    float(metrics.get("shortcut_mu_delta", 0.0)),
                    float(metrics.get("shortcut_std_delta", 0.0)),
                    float(metrics.get("shortcut_texture_gain", 0.0)),
                    float(metrics.get("shortcut_lsi", 0.0)),
                    float(metrics.get("train_num_steps", 0.0)),
                    float(metrics.get("train_step_size", 0.0)),
                    float(metrics.get("train_style_strength", 0.0)),
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
            "--clip_model_name",
            str(cfg_train.get("full_eval_clip_model_name", "openai/clip-vit-base-patch32")),
            "--clip_modelscope_id",
            str(cfg_train.get("full_eval_clip_modelscope_id", "")),
            "--clip_modelscope_cache_dir",
            str(cfg_train.get("full_eval_clip_modelscope_cache_dir", "")),
        ]
        if bool(cfg_train.get("full_eval_clip_allow_network", False)):
            cmd += ["--clip_allow_network"]
        full_eval_style_strength = cfg_train.get("full_eval_style_strength", cfg_infer.get("style_strength"))
        if full_eval_style_strength is not None:
            cmd += ["--style_strength", str(float(full_eval_style_strength))]
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
        if bool(cfg_train.get("full_eval_reuse_generated", True)):
            cmd += ["--reuse_generated"]
        if bool(cfg_train.get("full_eval_generation_only", False)):
            cmd += ["--generation_only"]

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
