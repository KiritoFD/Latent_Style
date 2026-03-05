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
    @staticmethod
    def _apply_channels_last_(module: torch.nn.Module) -> None:
        # Only 4D tensors support channels_last memory format.
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
        self._config_digest = self._compute_config_digest(config)

        train_cfg = config.get("training", {})
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
        self.cudnn_benchmark = bool(train_cfg.get("cudnn_benchmark", False))
        if torch.cuda.is_available():
            torch.backends.cuda.matmul.allow_tf32 = self.allow_tf32
            torch.backends.cudnn.allow_tf32 = self.allow_tf32
            torch.backends.cudnn.benchmark = self.cudnn_benchmark

        requested_channels_last = bool(train_cfg.get("channels_last", False) and device.type == "cuda")
        self.channels_last = bool(requested_channels_last)
        requested_empty_cache_interval = max(0, int(train_cfg.get("empty_cache_interval", 0)))
        self.empty_cache_interval = 0
        if requested_empty_cache_interval > 0:
            logger.warning(
                "empty_cache_interval=%d requested but forcibly disabled for allocator stability.",
                requested_empty_cache_interval,
            )
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
        self.model = self.model.to(device)
        if self.channels_last:
            self._apply_channels_last_(self.model)

        ckpt_cfg = config.get("checkpoint", {})
        self.checkpoint_dir = Path(ckpt_cfg.get("save_dir", "../adacut_ckpt"))
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir = self.checkpoint_dir / "logs"
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.profiler_dir = str(train_cfg.get("profiler_dir", str((self.log_dir / "profiler").resolve())))
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
            "Infra | channels_last=%s tf32=%s cudnn_benchmark=%s grad_ckpt=%s compile=%s backend=%s mode=%s fullgraph=%s cudagraphs=%s gc_collect_interval=%d loss_timing_interval=%d strict_batch_sanity=%s strict_batch_sanity_interval=%d cuda_sync_debug=%s profiler=%s alloc_conf=%s",
            self.channels_last,
            self.allow_tf32,
            self.cudnn_benchmark,
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
        amp_dtype_cfg = str(train_cfg.get("amp_dtype", "bf16")).lower()
        if amp_dtype_cfg in {"fp16", "float16", "half"}:
            self.amp_dtype = torch.float16
        else:
            self.amp_dtype = torch.bfloat16
        if device.type != "cuda":
            self.use_amp = False

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
            "Precision | amp=%s amp_dtype=%s fused_adamw=%s",
            self.use_amp,
            "fp16" if self.amp_dtype == torch.float16 else "bf16",
            use_fused_adamw,
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
                    "swd",
                    "color",
                    "identity",
                    "delta_tv",
                    "identity_ratio",
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

    @staticmethod
    def _compute_config_digest(config: Dict) -> str:
        try:
            return json.dumps(config, sort_keys=True, ensure_ascii=False)
        except Exception:
            return ""

    def _apply_loss_hot_config(self, config: Dict) -> Dict[str, tuple[float | int | str | list[int], float | int | str | list[int]]]:
        changed: Dict[str, tuple[float | int | str | list[int], float | int | str | list[int]]] = {}
        loss_cfg = config.get("loss", {})
        training_cfg = config.get("training", {})
        lf = self.loss_fn

        def _set_attr(name: str, value):
            old = getattr(lf, name)
            if old != value:
                setattr(lf, name, value)
                changed[f"loss.{name}"] = (old, value)

        patch_sizes = loss_cfg.get("swd_patch_sizes", [3, 5])
        if isinstance(patch_sizes, list):
            parsed_patch_sizes = [int(p) for p in patch_sizes if int(p) > 0]
        else:
            parsed_patch_sizes = [3, 5]
        if not parsed_patch_sizes:
            parsed_patch_sizes = [3, 5]

        _set_attr("w_swd", float(loss_cfg.get("w_swd", lf.w_swd)))
        _set_attr("swd_patch_sizes", parsed_patch_sizes)
        _set_attr("swd_num_projections", int(loss_cfg.get("swd_num_projections", lf.swd_num_projections)))
        _set_attr(
            "swd_projection_chunk_size",
            int(loss_cfg.get("swd_projection_chunk_size", getattr(lf, "swd_projection_chunk_size", 64))),
        )
        _set_attr(
            "swd_distance_mode",
            str(loss_cfg.get("swd_distance_mode", getattr(lf, "swd_distance_mode", "cdf"))).lower(),
        )
        _set_attr(
            "swd_cdf_num_bins",
            int(loss_cfg.get("swd_cdf_num_bins", getattr(lf, "swd_cdf_num_bins", 64))),
        )
        _set_attr(
            "swd_cdf_tau",
            float(loss_cfg.get("swd_cdf_tau", getattr(lf, "swd_cdf_tau", 0.01))),
        )
        _set_attr(
            "swd_cdf_sample_size",
            int(loss_cfg.get("swd_cdf_sample_size", getattr(lf, "swd_cdf_sample_size", 256))),
        )
        _set_attr(
            "swd_cdf_bin_chunk_size",
            int(loss_cfg.get("swd_cdf_bin_chunk_size", getattr(lf, "swd_cdf_bin_chunk_size", 16))),
        )
        _set_attr(
            "swd_cdf_sample_chunk_size",
            int(loss_cfg.get("swd_cdf_sample_chunk_size", getattr(lf, "swd_cdf_sample_chunk_size", 128))),
        )
        _set_attr("swd_batch_size", int(loss_cfg.get("swd_batch_size", lf.swd_batch_size)))
        _set_attr("swd_use_high_freq", bool(loss_cfg.get("swd_use_high_freq", lf.swd_use_high_freq)))
        _set_attr(
            "swd_hf_weight_ratio",
            float(loss_cfg.get("swd_hf_weight_ratio", getattr(lf, "swd_hf_weight_ratio", 2.0))),
        )
        _set_attr("w_color", float(loss_cfg.get("w_color", lf.w_color)))
        _set_attr("w_identity", float(loss_cfg.get("w_identity", lf.w_identity)))
        _set_attr("w_delta_tv", float(loss_cfg.get("w_delta_tv", lf.w_delta_tv)))
        _set_attr("nsight_nvtx", bool(training_cfg.get("nsight_nvtx", lf.nsight_nvtx)))
        if any(
            k in changed
            for k in (
                "loss.swd_patch_sizes",
                "loss.swd_num_projections",
                "loss.swd_distance_mode",
                "loss.swd_cdf_num_bins",
                "loss.swd_cdf_tau",
                "loss.swd_cdf_sample_size",
                "loss.swd_cdf_bin_chunk_size",
                "loss.swd_cdf_sample_chunk_size",
            )
        ):
            try:
                proj_cache = getattr(lf, "_projection_cache", None)
                if isinstance(proj_cache, dict):
                    proj_cache.clear()
            except Exception:
                pass

        return changed

    def reload_config_from_disk(self, epoch: int) -> bool:
        cfg_path_raw = self.config_path
        if not cfg_path_raw:
            return False
        cfg_path = Path(cfg_path_raw)
        if not cfg_path.is_absolute():
            cfg_path = (Path.cwd() / cfg_path).resolve()
        if not cfg_path.exists():
            logger.warning("Config hot-reload skipped at epoch %d: file not found: %s", epoch, cfg_path)
            return False

        try:
            with open(cfg_path, "r", encoding="utf-8") as f:
                new_config = json.load(f)
        except Exception as exc:
            logger.warning("Config hot-reload failed at epoch %d: %s", epoch, exc)
            return False

        new_digest = self._compute_config_digest(new_config)
        if new_digest == self._config_digest:
            return False

        changed: Dict[str, tuple[object, object]] = {}
        train_cfg = new_config.get("training", {})

        old_num_epochs = self.num_epochs
        self.num_epochs = int(train_cfg.get("num_epochs", self.num_epochs))
        if self.num_epochs != old_num_epochs:
            changed["training.num_epochs"] = (old_num_epochs, self.num_epochs)

        old_save_interval = self.save_interval
        self.save_interval = max(1, int(train_cfg.get("save_interval", self.save_interval)))
        if self.save_interval != old_save_interval:
            changed["training.save_interval"] = (old_save_interval, self.save_interval)

        old_full_eval_interval = self.full_eval_interval
        self.full_eval_interval = max(0, int(train_cfg.get("full_eval_interval", self.full_eval_interval)))
        if self.full_eval_interval != old_full_eval_interval:
            changed["training.full_eval_interval"] = (old_full_eval_interval, self.full_eval_interval)

        old_last_eval = self.run_full_eval_on_last_epoch
        self.run_full_eval_on_last_epoch = bool(train_cfg.get("full_eval_on_last_epoch", self.run_full_eval_on_last_epoch))
        if self.run_full_eval_on_last_epoch != old_last_eval:
            changed["training.full_eval_on_last_epoch"] = (old_last_eval, self.run_full_eval_on_last_epoch)

        old_grad_clip = self.grad_clip_norm
        self.grad_clip_norm = float(train_cfg.get("grad_clip_norm", self.grad_clip_norm))
        if self.grad_clip_norm != old_grad_clip:
            changed["training.grad_clip_norm"] = (old_grad_clip, self.grad_clip_norm)

        old_acc_steps = self.accumulation_steps
        self.accumulation_steps = max(1, int(train_cfg.get("accumulation_steps", self.accumulation_steps)))
        if self.accumulation_steps != old_acc_steps:
            changed["training.accumulation_steps"] = (old_acc_steps, self.accumulation_steps)

        old_log_interval = self.log_interval
        self.log_interval = max(0, int(train_cfg.get("log_interval", self.log_interval)))
        if self.log_interval != old_log_interval:
            changed["training.log_interval"] = (old_log_interval, self.log_interval)

        old_use_tqdm = self.use_tqdm
        self.use_tqdm = bool(train_cfg.get("use_tqdm", self.use_tqdm))
        if self.use_tqdm != old_use_tqdm:
            changed["training.use_tqdm"] = (old_use_tqdm, self.use_tqdm)

        requested_empty_cache_interval = max(0, int(train_cfg.get("empty_cache_interval", self.empty_cache_interval)))
        if requested_empty_cache_interval > 0:
            logger.warning(
                "Config hot-reload requested empty_cache_interval=%d but it remains disabled for allocator stability.",
                requested_empty_cache_interval,
            )

        old_log_vram_interval = self.log_vram_interval
        self.log_vram_interval = max(0, int(train_cfg.get("log_vram_interval", self.log_vram_interval)))
        if self.log_vram_interval != old_log_vram_interval:
            changed["training.log_vram_interval"] = (old_log_vram_interval, self.log_vram_interval)

        old_gc_collect_interval = self.gc_collect_interval
        self.gc_collect_interval = max(0, int(train_cfg.get("gc_collect_interval", self.gc_collect_interval)))
        if self.gc_collect_interval != old_gc_collect_interval:
            changed["training.gc_collect_interval"] = (old_gc_collect_interval, self.gc_collect_interval)

        old_trace_vram_steps = self.trace_vram_steps
        self.trace_vram_steps = max(0, int(train_cfg.get("trace_vram_steps", self.trace_vram_steps)))
        if self.trace_vram_steps != old_trace_vram_steps:
            changed["training.trace_vram_steps"] = (old_trace_vram_steps, self.trace_vram_steps)

        for group in self.optimizer.param_groups:
            old_lr = float(group.get("lr", 0.0))
            new_lr = float(train_cfg.get("learning_rate", old_lr))
            if old_lr != new_lr:
                group["lr"] = new_lr
                changed["training.learning_rate"] = (old_lr, new_lr)
                break
        for group in self.optimizer.param_groups:
            old_wd = float(group.get("weight_decay", 0.0))
            new_wd = float(train_cfg.get("weight_decay", old_wd))
            if old_wd != new_wd:
                group["weight_decay"] = new_wd
                changed["training.weight_decay"] = (old_wd, new_wd)
                break

        loss_changed = self._apply_loss_hot_config(new_config)
        changed.update(loss_changed)

        self.config = new_config
        self._config_digest = new_digest

        if changed:
            details = ", ".join(f"{k}: {v[0]} -> {v[1]}" for k, v in sorted(changed.items()))
            logger.info("Config hot-reloaded at epoch %d. Applied changes: %s", epoch, details)
        else:
            logger.info("Config file changed at epoch %d but no hot-reloadable fields differed.", epoch)
        return True

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
                    loss_to_backward.backward()
                backward_step += max(0.0, time.perf_counter() - t0)
                self._trace_vram(epoch=epoch, step_idx=step_idx, phase="after_backward")

                should_step = (step_idx % self.accumulation_steps == 0)
                if should_step:
                    t0 = time.perf_counter()
                    with self._nvtx_range("optimizer_step"):
                        if self.grad_clip_norm > 0:
                            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip_norm)
                        self.optimizer.step()
                        self.optimizer.zero_grad(set_to_none=True)
                    optimizer_step += max(0.0, time.perf_counter() - t0)
                    self._trace_vram(epoch=epoch, step_idx=step_idx, phase="after_optimizer_step")
                    self.global_step += 1
                    if self.gc_collect_interval > 0 and (self.global_step % self.gc_collect_interval == 0):
                        gc.collect()

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
                        swd=f"{_get_avg('swd'):.4f}",
                        color=f"{_get_avg('color'):.4f}",
                        idt=f"{_get_avg('identity'):.4f}",
                        dtv=f"{_get_avg('delta_tv'):.4f}",
                        idr=f"{_get_avg('identity_ratio'):.2f}",
                        data_ms=f"{(1000.0 * data_time_total / max(step_idx, 1)):.1f}",
                        comp_ms=f"{(1000.0 * compute_time_total / max(step_idx, 1)):.1f}",
                        it_s=f"{step_per_sec:.2f}",
                        eta=f"{eta:.1f}s",
                    )
                    if not self.use_tqdm:
                        logger.info(
                            "epoch %d step %d/%d | loss=%.4f swd=%.4f color=%.4f idt=%.4f dtv=%.4f idr=%.2f | data %.1fms comp %.1fms | %.2f it/s eta %.1fs",
                            epoch,
                            step_idx,
                            total_steps,
                            _get_avg('loss'),
                            _get_avg('swd'),
                            _get_avg('color'),
                            _get_avg('identity'),
                            _get_avg('delta_tv'),
                            _get_avg('identity_ratio'),
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
            if self.grad_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip_norm)
            self.optimizer.step()
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
            "loss", "swd", "color", "identity", "delta_tv", "identity_ratio",
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
                f"swd={metrics['swd']:.4f} "
                f"color={metrics['color']:.4f} "
                f"idt={metrics['identity']:.4f} dtv={metrics['delta_tv']:.4f} idr={metrics['identity_ratio']:.2f} "
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
                    float(metrics.get("swd", 0.0)),
                    float(metrics.get("color", 0.0)),
                    float(metrics.get("identity", 0.0)),
                    float(metrics.get("delta_tv", 0.0)),
                    float(metrics.get("identity_ratio", 0.0)),
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
            str(int(cfg_train.get("full_eval_batch_size", 20))),
            "--max_src_samples",
            str(int(cfg_train.get("full_eval_max_src_samples", 30))),
            "--max_ref_compare",
            str(int(cfg_train.get("full_eval_max_ref_compare", 50))),
            "--max_ref_cache",
            str(int(cfg_train.get("full_eval_max_ref_cache", 256))),
            "--ref_feature_batch_size",
            str(int(cfg_train.get("full_eval_ref_feature_batch_size", 128))),
            "--eval_lpips_chunk_size",
            str(int(cfg_train.get("full_eval_lpips_chunk_size", 8))),
            "--clip_model_name",
            str(cfg_train.get("full_eval_clip_model_name", "openai/clip-vit-base-patch32")),
            "--clip_modelscope_id",
            str(cfg_train.get("full_eval_clip_modelscope_id", "")),
            "--clip_modelscope_cache_dir",
            str(cfg_train.get("full_eval_clip_modelscope_cache_dir", "")),
            "--clip_hf_cache_dir",
            str(cfg_train.get("full_eval_clip_hf_cache_dir", "../eval_cache/hf")),
            "--image_classifier_path",
            str(cfg_train.get("full_eval_image_classifier_path", "../eval_cache/eval_style_image_classifier.pt")),
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
        if bool(cfg_train.get("full_eval_classifier_only", False)):
            cmd += ["--eval_classifier_only"]
        if bool(cfg_train.get("full_eval_disable_lpips", False)):
            cmd += ["--eval_disable_lpips"]
        if bool(cfg_train.get("full_eval_enable_art_fid", False)):
            cmd += ["--eval_enable_art_fid"]
            cmd += ["--eval_art_fid_max_gen", str(int(cfg_train.get("full_eval_art_fid_max_gen", 200)))]
            cmd += ["--eval_art_fid_max_ref", str(int(cfg_train.get("full_eval_art_fid_max_ref", 200)))]
            cmd += ["--eval_art_fid_batch_size", str(int(cfg_train.get("full_eval_art_fid_batch_size", 16)))]
            if bool(cfg_train.get("full_eval_art_fid_photo_only", False)):
                cmd += ["--eval_art_fid_photo_only"]
        if bool(cfg_train.get("full_eval_reuse_generated", True)):
            cmd += ["--reuse_generated"]
        if bool(cfg_train.get("full_eval_generation_only", False)):
            cmd += ["--generation_only"]

        log_path = self.log_dir / f"full_eval_epoch_{epoch:04d}.log"
        logger.info("Running full eval for epoch %d -> %s", epoch, out_dir)
        moved_to_cpu = False
        if self.device.type == "cuda":
            try:
                logger.info("Moving model to CPU to avoid VRAM swap during evaluation...")
                self.model = self.model.to("cpu")
                moved_to_cpu = True
            except Exception as exc:  # pragma: no cover
                logger.warning("Failed to move model to CPU before full eval, continue on current device: %s", exc)

        try:
            with open(log_path, "w", encoding="utf-8") as logf:
                proc = subprocess.run(cmd, cwd=str(Path(__file__).resolve().parent), stdout=logf, stderr=subprocess.STDOUT)
        finally:
            if moved_to_cpu:
                logger.info("Evaluation subprocess complete, moving model back to %s...", self.device)
                self.model = self.model.to(self.device)
                if self.channels_last:
                    self._apply_channels_last_(self.model)
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
        def _to_opt_float(v):
            if v is None:
                return None
            try:
                return float(v)
            except Exception:
                return None

        def _avg_opt(items, key: str):
            vals = [x.get(key) for x in items]
            vals = [v for v in vals if v is not None]
            if not vals:
                return None
            return float(sum(vals) / len(vals))

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
                    "transfer_clip_style": _to_opt_float(transfer.get("clip_style")),
                    "transfer_content_lpips": _to_opt_float(transfer.get("content_lpips")),
                    "transfer_fid": _to_opt_float(transfer.get("fid")),
                    "transfer_art_fid": _to_opt_float(transfer.get("art_fid")),
                    "transfer_classifier_acc": _to_opt_float(transfer.get("classifier_acc")),
                    "photo_to_art_clip_style": _to_opt_float(p2a.get("clip_style")),
                    "photo_to_art_fid": _to_opt_float(p2a.get("fid")),
                    "photo_to_art_art_fid": _to_opt_float(p2a.get("art_fid")),
                    "photo_to_art_classifier_acc": _to_opt_float(p2a.get("classifier_acc")),
                }
            )

        if not rounds:
            return

        rounds.sort(key=lambda x: x["epoch"])
        latest = rounds[-1]
        mean = {
            "transfer_clip_style": _avg_opt(rounds, "transfer_clip_style"),
            "transfer_content_lpips": _avg_opt(rounds, "transfer_content_lpips"),
            "transfer_fid": _avg_opt(rounds, "transfer_fid"),
            "transfer_art_fid": _avg_opt(rounds, "transfer_art_fid"),
            "transfer_classifier_acc": _avg_opt(rounds, "transfer_classifier_acc"),
            "photo_to_art_clip_style": _avg_opt(rounds, "photo_to_art_clip_style"),
            "photo_to_art_fid": _avg_opt(rounds, "photo_to_art_fid"),
            "photo_to_art_art_fid": _avg_opt(rounds, "photo_to_art_art_fid"),
            "photo_to_art_classifier_acc": _avg_opt(rounds, "photo_to_art_classifier_acc"),
        }

        rounds_with_transfer_fid = [x for x in rounds if x.get("transfer_fid") is not None]
        rounds_with_transfer_art_fid = [x for x in rounds if x.get("transfer_art_fid") is not None]
        rounds_with_photo_fid = [x for x in rounds if x.get("photo_to_art_fid") is not None]
        rounds_with_photo_art_fid = [x for x in rounds if x.get("photo_to_art_art_fid") is not None]
        rounds_with_transfer_cls = [x for x in rounds if x.get("transfer_classifier_acc") is not None]
        rounds_with_photo_cls = [x for x in rounds if x.get("photo_to_art_classifier_acc") is not None]

        best = {
            "best_transfer_classifier_acc": max(rounds_with_transfer_cls, key=lambda x: x["transfer_classifier_acc"]) if rounds_with_transfer_cls else None,
            "best_transfer_clip_style": max(rounds, key=lambda x: x["transfer_clip_style"] if x.get("transfer_clip_style") is not None else float("-inf")),
            "best_transfer_fid": min(rounds_with_transfer_fid, key=lambda x: x["transfer_fid"]) if rounds_with_transfer_fid else None,
            "best_transfer_art_fid": min(rounds_with_transfer_art_fid, key=lambda x: x["transfer_art_fid"]) if rounds_with_transfer_art_fid else None,
            "best_photo_to_art_classifier_acc": max(rounds_with_photo_cls, key=lambda x: x["photo_to_art_classifier_acc"]) if rounds_with_photo_cls else None,
            "best_photo_to_art_clip_style": max(rounds, key=lambda x: x["photo_to_art_clip_style"] if x.get("photo_to_art_clip_style") is not None else float("-inf")),
            "best_photo_to_art_fid": min(rounds_with_photo_fid, key=lambda x: x["photo_to_art_fid"]) if rounds_with_photo_fid else None,
            "best_photo_to_art_art_fid": min(rounds_with_photo_art_fid, key=lambda x: x["photo_to_art_art_fid"]) if rounds_with_photo_art_fid else None,
            "best_transfer_content_lpips": min(rounds, key=lambda x: x["transfer_content_lpips"] if x.get("transfer_content_lpips") is not None else float("inf")),
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
