from __future__ import annotations

import json
import logging
import os
import re
import threading
import time
import gc
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from config_schema import ExperimentConfig, compact_runtime_config
from losses import OTFlowMatchingObjective
from model import build_model_from_config, count_parameters
from style_families import prune_state_dict_for_tokenizer_family
from utils.training import (
    append_training_log,
    build_adamw,
    initialize_training_log,
    strip_compile_prefix,
    unwrap_compiled_model,
    write_config_and_source_snapshot,
)

logger = logging.getLogger(__name__)


def _style_cache_name(style_id: int, subdir: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(subdir)).strip("_") or f"style_{style_id}"
    return f"{style_id:02d}_{safe}.pt"


def _convert_4d_tensors_to_channels_last(module: torch.nn.Module) -> torch.nn.Module:
    """Keep tokenizer/vector parameters intact while using NHWC for conv tensors."""
    with torch.no_grad():
        for param in module.parameters():
            if param.is_floating_point() and param.ndim == 4:
                param.data = param.data.contiguous(memory_format=torch.channels_last)
                if param.grad is not None:
                    param.grad = param.grad.contiguous(memory_format=torch.channels_last)
        for buffer in module.buffers():
            if buffer.is_floating_point() and buffer.ndim == 4:
                buffer.data = buffer.data.contiguous(memory_format=torch.channels_last)
    return module


def _gradient_stats(x: torch.Tensor) -> torch.Tensor:
    dx = F.pad(x[..., :, 1:] - x[..., :, :-1], (0, 1, 0, 0))
    dy = F.pad(x[..., 1:, :] - x[..., :-1, :], (0, 0, 0, 1))
    mag = torch.sqrt(dx.square() + dy.square() + 1e-8)
    return torch.cat([mag.mean(dim=(2, 3)), mag.std(dim=(2, 3), unbiased=False)], dim=1)


def _host_can_resolve_path(path: Path) -> bool:
    try:
        text = str(path)
    except Exception:
        return False
    if os.name == "nt" and (
        text.startswith("/mnt/")
        or text.startswith("\\mnt\\")
        or text.startswith("/mnt\\")
        or text.startswith("\\mnt/")
    ):
        return False
    return True


def _move_tensor_tree(value, device: torch.device):
    if torch.is_tensor(value):
        return value.to(device=device, non_blocking=False)
    if isinstance(value, dict):
        for key, item in list(value.items()):
            value[key] = _move_tensor_tree(item, device)
        return value
    if isinstance(value, list):
        for idx, item in enumerate(list(value)):
            value[idx] = _move_tensor_tree(item, device)
        return value
    if isinstance(value, tuple):
        return tuple(_move_tensor_tree(item, device) for item in value)
    return value


def _latent_style_features(latents: torch.Tensor, *, dim: int, pool_size: int) -> torch.Tensor:
    x = latents.float().contiguous()
    low = F.avg_pool2d(x, kernel_size=5, stride=1, padding=2)
    high = x - low
    pool = max(1, int(pool_size))
    parts = [
        x.mean(dim=(2, 3)),
        x.std(dim=(2, 3), unbiased=False),
        low.std(dim=(2, 3), unbiased=False),
        high.std(dim=(2, 3), unbiased=False),
        high.abs().mean(dim=(2, 3)),
        _gradient_stats(x),
        F.adaptive_avg_pool2d(low, (pool, pool)).flatten(1),
        F.adaptive_avg_pool2d(high.abs(), (pool, pool)).flatten(1),
    ]
    fft_channels = min(2, int(x.shape[1]))
    if fft_channels > 0:
        fft_amp = torch.log(torch.fft.rfft2(high[:, :fft_channels], norm="ortho").abs() + 1e-8)
        parts.append(F.adaptive_avg_pool2d(fft_amp, (pool, max(1, pool // 2))).flatten(1))
    feat = torch.cat(parts, dim=1)
    feat = torch.nan_to_num(feat, nan=0.0, posinf=0.0, neginf=0.0)
    if feat.shape[1] < dim:
        feat = F.pad(feat, (0, dim - feat.shape[1]))
    elif feat.shape[1] > dim:
        feat = feat[:, :dim]
    return feat.contiguous()


def _kmeans_cpu(features: torch.Tensor, *, num_centers: int, iters: int) -> tuple[torch.Tensor, torch.Tensor]:
    n = int(features.shape[0])
    k = max(1, min(int(num_centers), n))
    if n == 0:
        raise ValueError("Cannot run k-means on an empty feature matrix")
    init_idx = torch.linspace(0, n - 1, steps=k).round().long()
    centers = features.index_select(0, init_idx).clone()
    assign = torch.zeros(n, dtype=torch.long)
    for _ in range(max(1, int(iters))):
        assign = torch.cdist(features, centers, p=2).argmin(dim=1)
        next_centers = centers.clone()
        for idx in range(k):
            mask = assign == idx
            if bool(mask.any()):
                next_centers[idx] = features[mask].mean(dim=0)
        centers = F.normalize(next_centers, p=2, dim=1, eps=1e-8)
    assign = torch.cdist(features, centers, p=2).argmin(dim=1)
    return centers, assign


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
            bridge_cfg=config.bridge,
            use_checkpointing=bool(train_cfg.get("use_gradient_checkpointing", False)),
        ).to(device)
        self._disable_legacy_style_components_for_pure_latent()
        self._maybe_initialize_tokenizer_from_latents()
        if self.channels_last:
            self.model = _convert_4d_tensors_to_channels_last(self.model)
        setattr(self.model, "profile_modules", bool(train_cfg.get("profile_modules", False)))
        setattr(self.model, "profile_sync_cuda", bool(train_cfg.get("profile_sync_cuda", False)))

        logger.info("Model params: %s", f"{count_parameters(self.model):,}")

        self.optimizer = self._build_optimizer([p for p in self.model.parameters() if p.requires_grad])

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
        self.stop_after_global_steps = max(0, int(train_cfg.get("stop_after_global_steps", 0)))
        raw_step_milestones = train_cfg.get("save_step_milestones", [])
        if isinstance(raw_step_milestones, str):
            raw_step_milestones = [part.strip() for part in raw_step_milestones.split(",") if part.strip()]
        self.save_step_milestones = sorted({int(v) for v in raw_step_milestones if int(v) > 0})
        self._saved_step_milestones: set[int] = set()
        self.async_checkpoint_save = bool(train_cfg.get("async_checkpoint_save", False))
        self._checkpoint_threads: list[threading.Thread] = []
        self.numeric_debug = bool(train_cfg.get("numeric_debug", False))
        self.numeric_debug_interval = max(1, int(train_cfg.get("numeric_debug_interval", 10)))
        self.numeric_debug_halt_on_nonfinite = bool(train_cfg.get("numeric_debug_halt_on_nonfinite", True))
        self.numeric_debug_dump_limit = max(1, int(train_cfg.get("numeric_debug_dump_limit", 200)))
        self.numeric_debug_events = 0
        self._offloaded_for_full_eval = False

        self.checkpoint_dir = Path(ckpt_cfg.save_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir = self.checkpoint_dir / "logs"
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.numeric_debug_file = self.checkpoint_dir / "numeric_debug.jsonl"

        write_config_and_source_snapshot(
            checkpoint_dir=self.checkpoint_dir,
            serialized_config=self.serialized_config,
            package_dir=Path(__file__).parent,
        )

        self.log_file = self.log_dir / f"training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        initialize_training_log(self.log_file)

        self.global_step = 0
        self.requested_stop = False
        self.start_epoch = 1
        self._maybe_resume(str(train_cfg.get("resume_checkpoint", "")))
        self._configure_freeze_mode()
        self._configure_distillation()
        self._configure_compile()

    def _move_optimizer_state(self, device: torch.device) -> None:
        for state in self.optimizer.state.values():
            _move_tensor_tree(state, device)

    def _pure_latent_uses_structured_tokenizer(self) -> bool:
        return str(getattr(self.config.model, "tokenizer_family", "legacy_factorized")).strip().lower() == "pure_latent_spatial"

    def _style_branch_uses_legacy_spatial_priors(self) -> bool:
        if self._pure_latent_uses_structured_tokenizer():
            return False
        return not bool(getattr(self.config.model, "ablation_disable_spatial_prior", False))

    def _disable_legacy_style_components_for_pure_latent(self) -> None:
        if not self._pure_latent_uses_structured_tokenizer():
            return
        tokenizer = getattr(self.model, "style_tokenizer", None)
        if tokenizer is not None:
            for _, param in tokenizer.named_parameters():
                param.requires_grad_(False)
        legacy_spatial = getattr(self.model, "style_spatial_id_16", None)
        if isinstance(legacy_spatial, torch.nn.Parameter):
            legacy_spatial.requires_grad_(False)
        for extra_name in ("style_spatial_proto_16", "style_spatial_atoms_16"):
            value = getattr(self.model, extra_name, None)
            if isinstance(value, torch.nn.Parameter):
                value.requires_grad_(False)
        logits = getattr(self.model, "style_spatial_logits", None)
        if logits is not None:
            for _, param in logits.named_parameters():
                param.requires_grad_(False)
        router = getattr(self.model, "style_spatial_content_router", None)
        if router is not None:
            for _, param in router.named_parameters():
                param.requires_grad_(False)

    def offload_for_full_eval(self) -> None:
        if self.device.type != "cuda" or self._offloaded_for_full_eval:
            return
        logger.info("Offloading training state to CPU before remote full eval.")
        clear_model = getattr(self.model, "clear_runtime_caches", None)
        if callable(clear_model):
            clear_model()
        clear_teacher = getattr(self.teacher_model, "clear_runtime_caches", None)
        if callable(clear_teacher):
            clear_teacher()
        if self.teacher_model is not None:
            self.teacher_model.to("cpu")
        self.model.to("cpu")
        self._move_optimizer_state(torch.device("cpu"))
        gc.collect()
        torch.cuda.empty_cache()
        self._offloaded_for_full_eval = True

    def restore_after_full_eval(self) -> None:
        if self.device.type != "cuda" or not self._offloaded_for_full_eval:
            return
        logger.info("Restoring training state to CUDA after remote full eval.")
        self.model.to(self.device)
        if self.channels_last:
            self.model = _convert_4d_tensors_to_channels_last(self.model)
        self._move_optimizer_state(self.device)
        if self.teacher_model is not None:
            self.teacher_model.to(self.device)
            if self.channels_last:
                self.teacher_model = _convert_4d_tensors_to_channels_last(self.teacher_model)
        gc.collect()
        torch.cuda.empty_cache()
        self._offloaded_for_full_eval = False

    def _maybe_initialize_tokenizer_from_latents(self) -> None:
        model_cfg = self.config.model
        mode = str(getattr(model_cfg, "tokenizer_latent_init_mode", "none") or "none").strip().lower()
        if mode in {"", "none", "off", "false", "0"}:
            return
        if self._pure_latent_uses_structured_tokenizer():
            logger.info("Skipping legacy tokenizer latent init because tokenizer_family=pure_latent_spatial uses structured_style_tokenizer as the active path.")
            return
        tokenizer = getattr(self.model, "style_tokenizer", None)
        if tokenizer is None:
            logger.warning("tokenizer_latent_init_mode=%s requested, but model has no style_tokenizer.", mode)
            return
        data_cfg = self.config.data
        style_subdirs = list(data_cfg.style_subdirs)
        cache_dir = str(getattr(model_cfg, "tokenizer_latent_init_cache_dir", "") or "").strip()
        latent_cache_dir = Path(cache_dir) if cache_dir else Path(data_cfg.latent_cache_dir or "") / "packed"
        if not cache_dir:
            latent_cache_dir = (Path(data_cfg.data_root) / ".latent_cache" / "packed") if not data_cfg.latent_cache_dir else (Path(data_cfg.latent_cache_dir) / "packed")
        elif (latent_cache_dir / "packed").exists():
            latent_cache_dir = latent_cache_dir / "packed"
        if not _host_can_resolve_path(latent_cache_dir):
            logger.info(
                "Skipping tokenizer latent init on this host because packed cache path is WSL-only: %s",
                latent_cache_dir,
            )
            return
        if not latent_cache_dir.exists():
            logger.warning("Skipping tokenizer latent init: packed cache missing at %s", latent_cache_dir)
            return
        try:
            features_by_style = self._load_latent_init_features(latent_cache_dir, style_subdirs)
            self._apply_tokenizer_latent_init(tokenizer, features_by_style)
        except Exception:
            logger.exception("Tokenizer latent init failed; continuing with random tokenizer initialization.")

    def _load_latent_init_features(self, packed_dir: Path, style_subdirs: list[str]) -> list[torch.Tensor]:
        dim = int(self.config.model.style_dim)
        pool_size = int(getattr(self.config.model, "tokenizer_latent_init_pool_size", 4))
        sample_limit = int(getattr(self.config.model, "tokenizer_latent_init_sample_limit_per_style", 1000))
        features: list[torch.Tensor] = []
        for style_id, subdir in enumerate(style_subdirs):
            path = packed_dir / _style_cache_name(style_id, subdir)
            payload = torch.load(path, map_location="cpu", weights_only=False)
            if not isinstance(payload, dict) or not torch.is_tensor(payload.get("latents")):
                raise ValueError(f"Invalid packed latent cache: {path}")
            latents = payload["latents"].float()
            if sample_limit > 0 and latents.shape[0] > sample_limit:
                idx = torch.linspace(0, latents.shape[0] - 1, steps=sample_limit).round().long()
                latents = latents.index_select(0, idx)
            features.append(_latent_style_features(latents, dim=dim, pool_size=pool_size))
        all_features = torch.cat(features, dim=0)
        mean = all_features.mean(dim=0, keepdim=True)
        std = all_features.std(dim=0, keepdim=True, unbiased=False).clamp_min(1e-6)
        return [F.normalize((feat - mean) / std, p=2, dim=1, eps=1e-8) for feat in features]

    def _apply_tokenizer_latent_init(self, tokenizer: torch.nn.Module, features_by_style: list[torch.Tensor]) -> None:
        projection_mode = str(getattr(tokenizer, "projection_mode", "")).strip().lower()
        style_dim = int(getattr(tokenizer, "style_dim", self.config.model.style_dim))
        scale = float(getattr(self.config.model, "tokenizer_latent_init_scale", 0.2)) * (style_dim ** 0.5)
        kmeans_iters = int(getattr(self.config.model, "tokenizer_latent_init_kmeans_iters", 8))
        style_means = torch.stack([F.normalize(feat.mean(dim=0, keepdim=True), p=2, dim=1, eps=1e-8).squeeze(0) for feat in features_by_style])
        style_means = style_means * scale
        device = next(tokenizer.parameters()).device
        dtype = next(tokenizer.parameters()).dtype

        with torch.no_grad():
            if projection_mode == "class_prototypes" and hasattr(tokenizer, "class_prototypes"):
                k = int(getattr(tokenizer, "num_prototypes", 1))
                proto_rows: list[torch.Tensor] = []
                logits_rows: list[torch.Tensor] = []
                for feat in features_by_style:
                    centers, assign = _kmeans_cpu(feat, num_centers=k, iters=kmeans_iters)
                    if centers.shape[0] < k:
                        centers = torch.cat([centers, centers[-1:].expand(k - centers.shape[0], -1)], dim=0)
                    counts = torch.bincount(assign, minlength=k).float().clamp_min(1.0)
                    probs = counts / counts.sum()
                    proto_rows.append(centers[:k] * scale)
                    logits_rows.append(torch.log(probs) * float(getattr(tokenizer, "atom_temperature", 1.0)))
                tokenizer.class_prototypes.copy_(torch.stack(proto_rows).to(device=device, dtype=dtype))
                tokenizer.prototype_logits.weight.copy_(torch.stack(logits_rows).to(device=device, dtype=dtype))
                logger.info("Initialized class_prototypes tokenizer from VAE latent statistics.")
                return

            if projection_mode in {"concept_atoms", "direct_atom_residual", "global_vq"} and hasattr(tokenizer, "concept_atoms"):
                atom_count = int(getattr(tokenizer, "num_atoms", tokenizer.concept_atoms.shape[0]))
                all_features = torch.cat(features_by_style, dim=0)
                atoms, _ = _kmeans_cpu(all_features, num_centers=atom_count, iters=kmeans_iters)
                if atoms.shape[0] < atom_count:
                    atoms = torch.cat([atoms, atoms[-1:].expand(atom_count - atoms.shape[0], -1)], dim=0)
                atoms = atoms[:atom_count] * scale
                dist = torch.cdist(F.normalize(style_means / max(scale, 1e-8), p=2, dim=1), F.normalize(atoms / max(scale, 1e-8), p=2, dim=1), p=2)
                logits = -dist * float(getattr(tokenizer, "atom_temperature", 1.0))
                weights = F.softmax(logits / max(float(getattr(tokenizer, "atom_temperature", 1.0)), 1e-6), dim=-1)
                tokenizer.concept_atoms.copy_(atoms.to(device=device, dtype=dtype))
                tokenizer.atom_logits.weight.copy_(logits.to(device=device, dtype=dtype))
                if projection_mode == "direct_atom_residual" and hasattr(tokenizer, "direct_code"):
                    residual = weights @ atoms
                    direct = style_means - float(getattr(tokenizer, "residual_gain", 0.0)) * residual
                    tokenizer.direct_code.weight.copy_(direct.to(device=device, dtype=dtype))
                logger.info("Initialized %s tokenizer from VAE latent statistics.", projection_mode)
                return

            if projection_mode == "direct_code" and hasattr(tokenizer, "direct_code"):
                tokenizer.direct_code.weight.copy_(style_means.to(device=device, dtype=dtype))
                logger.info("Initialized direct_code tokenizer from VAE latent statistics.")
                return

        logger.warning("tokenizer_latent_init_mode requested but unsupported for projection_mode=%s.", projection_mode)

    def _snapshot_for_checkpoint(self, value):
        if torch.is_tensor(value):
            return value.detach().cpu().clone()
        if isinstance(value, dict):
            return {key: self._snapshot_for_checkpoint(val) for key, val in value.items()}
        if isinstance(value, list):
            return [self._snapshot_for_checkpoint(val) for val in value]
        if isinstance(value, tuple):
            return tuple(self._snapshot_for_checkpoint(val) for val in value)
        return value

    def _prune_checkpoint_threads(self) -> None:
        self._checkpoint_threads = [thread for thread in self._checkpoint_threads if thread.is_alive()]

    def wait_for_pending_checkpoints(self) -> None:
        for thread in self._checkpoint_threads:
            thread.join()
        self._checkpoint_threads.clear()

    def _configure_compile(self) -> None:
        if not bool(self.train_cfg.get("torch_compile", False)):
            return
        if not hasattr(torch, "compile"):
            logger.warning("training.torch_compile requested, but this PyTorch build has no torch.compile.")
            return

        cache_dir_raw = str(self.train_cfg.get("torch_compile_cache_dir", "")).strip()
        if cache_dir_raw:
            cache_dir = Path(cache_dir_raw)
            if not cache_dir.is_absolute():
                cache_dir = (Path.cwd() / cache_dir).resolve()
            inductor_dir = cache_dir / "inductor"
            triton_dir = cache_dir / "triton"
            inductor_dir.mkdir(parents=True, exist_ok=True)
            triton_dir.mkdir(parents=True, exist_ok=True)
            os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR", str(inductor_dir))
            os.environ.setdefault("TRITON_CACHE_DIR", str(triton_dir))

        backend = str(self.train_cfg.get("torch_compile_backend", "inductor") or "inductor")
        mode_raw = str(self.train_cfg.get("torch_compile_mode", "default") or "default").strip()
        mode = None if mode_raw in {"", "default", "none"} else mode_raw
        fullgraph = bool(self.train_cfg.get("torch_compile_fullgraph", False))
        dynamic = self.train_cfg.get("torch_compile_dynamic", None)
        if dynamic is not None:
            dynamic = bool(dynamic)

        logger.info(
            "Compiling model with torch.compile backend=%s mode=%s fullgraph=%s dynamic=%s cache=%s",
            backend,
            mode or "default",
            fullgraph,
            dynamic,
            cache_dir_raw or "<default>",
        )
        try:
            self.model = torch.compile(
                self.model,
                backend=backend,
                mode=mode,
                fullgraph=fullgraph,
                dynamic=dynamic,
            )
        except Exception:
            logger.exception("torch.compile failed during setup.")
            raise

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

    def _tokenizer_debug_stats(self) -> Dict[str, float]:
        stats: Dict[str, float] = {}
        modules = [
            ("style_tokenizer", getattr(self.model, "style_tokenizer", None)),
            ("structured_style_tokenizer", getattr(self.model, "structured_style_tokenizer", None)),
        ]
        scalar_stats = self._tokenizer_scalar_metrics()
        if scalar_stats:
            stats.update(scalar_stats)
        for prefix, module in modules:
            if module is None:
                continue
            for name, param in module.named_parameters():
                grad = param.grad
                if grad is not None:
                    stats[f"{prefix}_grad_{name.replace('.', '_')}"] = float(
                        torch.nan_to_num(grad.detach().float().abs().mean()).item()
                    )
        return stats

    def _tokenizer_scalar_metrics(self) -> Dict[str, float]:
        stats: Dict[str, float] = {}
        modules = [
            ("style_tokenizer", getattr(self.model, "style_tokenizer", None)),
            ("structured_style_tokenizer", getattr(self.model, "structured_style_tokenizer", None)),
        ]
        for prefix, module in modules:
            if module is None:
                continue
            raw = getattr(module, "last_debug", {})
            if isinstance(raw, dict):
                for key, value in raw.items():
                    if torch.is_tensor(value):
                        stats[f"{prefix}_{key}"] = float(torch.nan_to_num(value.detach().float()).item())
                    elif isinstance(value, (int, float, bool)):
                        stats[f"{prefix}_{key}"] = float(value)
        return stats

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
            "semantic_topology_attn": self._tensor_stats(getattr(self.model, "last_semantic_topology_attn", None)),
        }
        if extra:
            payload["extra"] = extra
        token_stats = self._tokenizer_debug_stats()
        if token_stats:
            payload["style_tokenizer"] = token_stats
        self.numeric_debug_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self.numeric_debug_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")
        self.numeric_debug_events += 1

    def _build_optimizer(self, params) -> torch.optim.Optimizer:
        return build_adamw(params, self.train_cfg, self.device)

    def _rebuild_scheduler_for_current_optimizer(self) -> None:
        if self.scheduler is None:
            return
        if self.scheduler_name == "multistep":
            milestones = sorted(int(v) for v in self.train_cfg.get("multistep_milestones", [40, 55]))
            gamma = float(self.train_cfg.get("multistep_gamma", 0.1))
            self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=milestones, gamma=gamma)
            return
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=max(1, int(self.train_cfg.get("num_epochs", 60))),
            eta_min=float(self.train_cfg.get("min_learning_rate", 5e-5)),
        )

    def _find_latest_checkpoint(self) -> Optional[Path]:
        ckpts = sorted(self.checkpoint_dir.glob("epoch_*.pt"))
        return ckpts[-1] if ckpts else None

    def _maybe_resume(self, resume_checkpoint: str) -> None:
        explicit_ckpt = None
        if resume_checkpoint:
            explicit_ckpt = Path(resume_checkpoint)
            if not explicit_ckpt.is_absolute():
                explicit_ckpt = (Path.cwd() / explicit_ckpt).resolve()
        local_latest = self._find_latest_checkpoint()
        prefer_local_latest = bool(self.train_cfg.get("resume_prefer_local_checkpoint", True))
        using_local_latest = False
        if prefer_local_latest and local_latest is not None:
            ckpt_path = local_latest
            using_local_latest = True
            if explicit_ckpt is not None and explicit_ckpt != local_latest:
                logger.info(
                    "Preferring local run checkpoint over configured resume target: local=%s configured=%s",
                    local_latest,
                    explicit_ckpt,
                )
        else:
            ckpt_path = explicit_ckpt
        if ckpt_path is None or not ckpt_path.exists():
            logger.info("No checkpoint found, start from scratch.")
            return
        state = torch.load(ckpt_path, map_location=self.device, weights_only=False)
        model_state = strip_compile_prefix(state["model_state_dict"])
        model_state, removed_contract_keys = prune_state_dict_for_tokenizer_family(
            model_state,
            tokenizer_family=str(getattr(self.config.model, "tokenizer_family", "legacy_factorized")),
            style_injection_mode=str(getattr(self.config.model, "style_injection_mode", "none")),
            proximal_mode=str(getattr(self.config.model, "proximal_mode", "off")),
        )
        if removed_contract_keys:
            logger.info(
                "Pruned %d legacy contract keys while resuming %s for tokenizer_family=%s",
                len(removed_contract_keys),
                ckpt_path,
                str(getattr(self.config.model, "tokenizer_family", "legacy_factorized")),
            )
        resume_model_strict = bool(self.train_cfg.get("resume_model_strict", True))
        ignore_prefixes = tuple(str(v) for v in self.train_cfg.get("resume_ignore_prefixes", []) if str(v))
        include_prefixes = tuple(str(v) for v in self.train_cfg.get("resume_include_prefixes", []) if str(v))
        if resume_model_strict and not ignore_prefixes and not include_prefixes:
            self.model.load_state_dict(model_state, strict=True)
        else:
            current = self.model.state_dict()
            compatible = {}
            skipped = []
            for key, value in model_state.items():
                if include_prefixes and not key.startswith(include_prefixes):
                    skipped.append(key)
                    continue
                if ignore_prefixes and key.startswith(ignore_prefixes):
                    skipped.append(key)
                    continue
                if key not in current or current[key].shape != value.shape:
                    skipped.append(key)
                    continue
                compatible[key] = value
            missing, unexpected = self.model.load_state_dict(compatible, strict=False)
            logger.info(
                "Partially resumed model from %s | loaded=%d skipped=%d missing=%d unexpected=%d include=%s ignore=%s",
                ckpt_path,
                len(compatible),
                len(skipped),
                len(missing),
                len(unexpected),
                list(include_prefixes),
                list(ignore_prefixes),
            )
        resume_optimizer = bool(self.train_cfg.get("resume_optimizer", True) or using_local_latest)
        if resume_optimizer and "optimizer_state_dict" in state:
            try:
                self.optimizer.load_state_dict(state["optimizer_state_dict"])
            except ValueError as exc:
                logger.warning(
                    "Skipping optimizer resume for %s due to state mismatch: %s",
                    ckpt_path,
                    exc,
                )
                resume_optimizer = False
        if resume_optimizer and self.scheduler is not None and state.get("scheduler_state_dict") is not None:
            try:
                self.scheduler.load_state_dict(state["scheduler_state_dict"])
            except ValueError as exc:
                logger.warning(
                    "Skipping scheduler resume for %s due to state mismatch: %s",
                    ckpt_path,
                    exc,
                )
        if "loss_state_dict" in state and hasattr(self.loss_fn, "load_state_dict"):
            self.loss_fn.load_state_dict(state.get("loss_state_dict"))
        if bool(self.train_cfg.get("resume_training_state", True) or using_local_latest):
            self.global_step = int(state.get("global_step", 0))
            self.start_epoch = int(state.get("epoch", 0)) + 1
        logger.info("Resumed from %s at epoch=%d global_step=%d", ckpt_path, self.start_epoch, self.global_step)

    def _reset_trainable_style_params(self, mode: str) -> None:
        with torch.no_grad():
            tokenizer = getattr(self.model, "style_tokenizer", None)
            if mode in {"tokenizer_only", "style_branch"} and tokenizer is not None and not self._pure_latent_uses_structured_tokenizer():
                tokenizer.reset_parameters()
            legacy_spatial = getattr(self.model, "style_spatial_id_16", None)
            if mode == "style_branch" and isinstance(legacy_spatial, torch.nn.Parameter):
                torch.nn.init.normal_(legacy_spatial, mean=0.0, std=0.02)

    def _configure_freeze_mode(self) -> None:
        if self.distill_enabled:
            return
        mode = str(self.train_cfg.get("freeze_mode", "none")).strip().lower()
        if mode in {"", "none", "all"}:
            return
        aliases = {
            "style_tokenizer_only": "tokenizer_only",
            "token_only": "tokenizer_only",
            "tokenizer_branch": "style_branch",
            "lancet_only": "backbone_only",
            "consumer_only": "backbone_only",
            "freeze_tokenizer": "backbone_only",
            "body_attention_only": "attention_only",
            "renderer_only": "executor_only",
            "fresh_executor": "executor_only",
            "freeze_style_branch": "executor_only",
            "execution_budget_only": "budget_only",
            "budget_branch": "budget_only",
            "style_injection_only": "injection_only",
            "injection_branch": "injection_only",
        }
        mode = aliases.get(mode, mode)
        if mode not in {"tokenizer_only", "style_branch", "backbone_only", "attention_only", "executor_only", "budget_only", "injection_only"}:
            raise ValueError(f"Unsupported freeze_mode: {mode}")

        for _, param in self.model.named_parameters():
            param.requires_grad_(False)

        trainable_names: list[str] = []
        if mode in {"tokenizer_only", "style_branch"}:
            tokenizer = getattr(self.model, "style_tokenizer", None)
            if tokenizer is not None and not self._pure_latent_uses_structured_tokenizer():
                for name, param in tokenizer.named_parameters():
                    param.requires_grad_(True)
                    trainable_names.append(f"style_tokenizer.{name}")
            structured = getattr(self.model, "structured_style_tokenizer", None)
            if structured is not None:
                for name, param in structured.named_parameters():
                    param.requires_grad_(True)
                    trainable_names.append(f"structured_style_tokenizer.{name}")
        if mode == "style_branch" and self._style_branch_uses_legacy_spatial_priors():
            self.model.style_spatial_id_16.requires_grad_(True)
            trainable_names.append("style_spatial_id_16")
            for extra_name in ("style_spatial_proto_16", "style_spatial_atoms_16"):
                value = getattr(self.model, extra_name, None)
                if isinstance(value, torch.nn.Parameter):
                    value.requires_grad_(True)
                    trainable_names.append(extra_name)
            logits = getattr(self.model, "style_spatial_logits", None)
            if logits is not None:
                for name, param in logits.named_parameters():
                    param.requires_grad_(True)
                    trainable_names.append(f"style_spatial_logits.{name}")
            router = getattr(self.model, "style_spatial_content_router", None)
            if router is not None:
                for name, param in router.named_parameters():
                    param.requires_grad_(True)
                    trainable_names.append(f"style_spatial_content_router.{name}")
        if mode == "budget_only":
            budget_head = getattr(self.model, "execution_budget_head", None)
            if budget_head is None:
                raise RuntimeError("freeze_mode=budget_only requires model.execution_budget_mode != 'none'.")
            for name, param in budget_head.named_parameters():
                param.requires_grad_(True)
                trainable_names.append(f"execution_budget_head.{name}")
        if mode == "injection_only":
            injectors = [
                ("body_style_injector", getattr(self.model, "body_style_injector", None)),
                ("decoder_style_injector", getattr(self.model, "decoder_style_injector", None)),
                ("body_style_carrier", getattr(self.model, "body_style_carrier", None)),
                ("body_content_gate", getattr(self.model, "body_content_gate", None)),
                ("decoder_style_carrier", getattr(self.model, "decoder_style_carrier", None)),
                ("decoder_content_gate", getattr(self.model, "decoder_content_gate", None)),
                ("body_style_spatial_proj", getattr(self.model, "body_style_spatial_proj", None)),
                ("body_structure_gate", getattr(self.model, "body_structure_gate", None)),
                ("decoder_style_spatial_proj", getattr(self.model, "decoder_style_spatial_proj", None)),
                ("decoder_structure_gate", getattr(self.model, "decoder_structure_gate", None)),
            ]
            for prefix, module in injectors:
                if module is None:
                    continue
                for name, param in module.named_parameters():
                    param.requires_grad_(True)
                    trainable_names.append(f"{prefix}.{name}")
            if not trainable_names:
                raise RuntimeError("freeze_mode=injection_only requires model.style_injection_mode != 'none'.")
        if mode == "backbone_only":
            for name, param in self.model.named_parameters():
                if name.startswith("style_tokenizer."):
                    continue
                if name.startswith("structured_style_tokenizer."):
                    continue
                param.requires_grad_(True)
                trainable_names.append(name)
        if mode == "attention_only":
            for name, param in self.model.named_parameters():
                if name.startswith("body_blocks."):
                    param.requires_grad_(True)
                    trainable_names.append(name)
                    continue
                if name.startswith("blender."):
                    param.requires_grad_(True)
                    trainable_names.append(name)
        if mode == "executor_only":
            for name, param in self.model.named_parameters():
                if name.startswith("style_tokenizer.") or name.startswith("structured_style_tokenizer.") or name == "style_spatial_id_16":
                    continue
                param.requires_grad_(True)
                trainable_names.append(name)

        if bool(self.train_cfg.get("freeze_reinit_trainable", False)):
            self._reset_trainable_style_params(mode)

        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        if not trainable_params:
            raise RuntimeError(f"freeze_mode={mode} selected no trainable parameters.")
        self.optimizer = self._build_optimizer(trainable_params)
        self._rebuild_scheduler_for_current_optimizer()
        logger.info("Freeze mode=%s | trainable_count=%d | trainable=%s", mode, len(trainable_params), ", ".join(trainable_names[:24]))

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
        teacher_state, removed_contract_keys = prune_state_dict_for_tokenizer_family(
            teacher_state,
            tokenizer_family=str(getattr(self.config.model, "tokenizer_family", "legacy_factorized")),
            style_injection_mode=str(getattr(self.config.model, "style_injection_mode", "none")),
            proximal_mode=str(getattr(self.config.model, "proximal_mode", "off")),
        )
        if removed_contract_keys:
            logger.info(
                "Pruned %d legacy contract keys from distillation teacher %s for tokenizer_family=%s",
                len(removed_contract_keys),
                teacher_ckpt,
                str(getattr(self.config.model, "tokenizer_family", "legacy_factorized")),
            )
        if not str(self.train_cfg.get("resume_checkpoint", "")).strip():
            self.model.load_state_dict(teacher_state, strict=True)

        teacher = build_model_from_config(
            self.config.model,
            bridge_cfg=self.config.bridge,
            use_checkpointing=False,
        ).to(self.device)
        if self.channels_last:
            teacher = _convert_4d_tensors_to_channels_last(teacher)
        teacher.load_state_dict(teacher_state, strict=True)
        teacher.eval()
        for param in teacher.parameters():
            param.requires_grad_(False)
        self.teacher_model = teacher

        mode = str(self.distill_cfg.get("mode", "tokenizer_only")).strip().lower()
        if mode not in {"tokenizer_only", "style_branch"}:
            raise ValueError(f"Unsupported distill mode: {mode}")

        for _, param in self.model.named_parameters():
            param.requires_grad_(False)

        trainable_names: list[str] = []
        if mode in {"tokenizer_only", "style_branch"}:
            tokenizer = getattr(self.model, "style_tokenizer", None)
            if tokenizer is not None and not self._pure_latent_uses_structured_tokenizer():
                for name, param in tokenizer.named_parameters():
                    param.requires_grad_(True)
                    trainable_names.append(f"style_tokenizer.{name}")
            structured = getattr(self.model, "structured_style_tokenizer", None)
            if structured is not None:
                for name, param in structured.named_parameters():
                    param.requires_grad_(True)
                    trainable_names.append(f"structured_style_tokenizer.{name}")
        if mode == "style_branch" and self._style_branch_uses_legacy_spatial_priors():
            self.model.style_spatial_id_16.requires_grad_(True)
            trainable_names.append("style_spatial_id_16")

        if bool(self.distill_cfg.get("reinit_trainable", True)):
            self._reset_trainable_style_params(mode)

        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        if not trainable_params:
            raise RuntimeError("Distillation enabled but no trainable parameters were selected.")
        self.optimizer = self._build_optimizer(trainable_params)
        if self.scheduler is not None:
            self._rebuild_scheduler_for_current_optimizer()
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
        setattr(self.model, "current_epoch", int(epoch))
        setattr(self.model, "total_epochs", int(self.num_epochs))
        self.model.train()
        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.device)
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
            aux_target_style = batch.get("aux_target_style")
            aux_target_valid = batch.get("aux_target_valid")

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
                        conditioning=batch,
                    )
                else:
                    loss_dict = self.loss_fn.compute(
                        self.model,
                        content=content,
                        target_style=target_style,
                        target_style_id=target_style_id,
                        source_style_id=source_style_id,
                        aux_target_style=aux_target_style,
                        aux_target_valid=aux_target_valid,
                        conditioning=batch,
                    )
                loss = loss_dict["loss"]
            forward_time_total += max(0.0, time.perf_counter() - t0)
            should_debug_step = self.numeric_debug and (
                step_idx == 1 or step_idx % self.numeric_debug_interval == 0
            )
            loss_is_finite = True
            if should_debug_step:
                loss_is_finite = bool(torch.isfinite(loss.detach()).item())
                self._write_numeric_debug(
                    epoch=epoch,
                    step=step_idx,
                    stage="forward",
                    loss_dict=loss_dict,
                    target_style_id=target_style_id,
                    source_style_id=source_style_id,
                )
            if should_debug_step and not loss_is_finite:
                msg = f"Non-finite loss detected at epoch={epoch} step={step_idx}"
                logger.error(msg)
                if self.numeric_debug_halt_on_nonfinite:
                    raise FloatingPointError(msg)

            t0 = time.perf_counter()
            (loss / self.accumulation_steps).backward()
            backward_time_total += max(0.0, time.perf_counter() - t0)
            grad_report = None
            has_nonfinite_grad = False
            if should_debug_step:
                grad_report = self._grad_stats()
                has_nonfinite_grad = bool(grad_report["first_nonfinite_grad_name"] is not None)
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
                milestone_metrics = {
                    key: float((value / max(num_batches + 1, 1)).item())
                    for key, value in metric_accum.items()
                }
                milestone_metrics["loss"] = float(loss.detach().item())
                milestone_metrics["lr"] = float(self.optimizer.param_groups[0]["lr"])
                self.maybe_save_step_checkpoint(epoch, step_idx, milestone_metrics)
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
            token_scalar_metrics = self._tokenizer_scalar_metrics()
            if token_scalar_metrics:
                for key, value in token_scalar_metrics.items():
                    scalar = content.new_tensor(float(value), dtype=torch.float32)
                    metric_accum[key] = metric_accum.get(key, 0) + scalar
            num_batches += 1

            compute_time_total = forward_time_total + backward_time_total + optimizer_time_total
            progress_interval = max(1, self.log_interval)
            if self.use_tqdm and (
                step_idx == 1 or step_idx % progress_interval == 0 or step_idx == len(dataloader)
            ):
                postfix = {
                    "loss": f"{_avg('loss'):.4f}",
                    "flow": f"{_avg('flow'):.4f}" if not self.distill_enabled else f"{_avg('distill_velocity'):.4f}",
                    "kin": f"{_avg('kinetic_energy'):.4f}",
                    "curv": f"{_avg('curvature'):.4f}",
                    "ot": f"{_avg('ot_cost'):.4f}",
                    "tswd": f"{_avg('terminal_swd'):.4f}" if not self.distill_enabled else f"{_avg('distill_endpoint'):.4f}",
                    "t": f"{_avg('t_mean'):.3f}",
                }
                if "content_lowpass_anchor" in metric_accum or "content_edge_anchor" in metric_accum:
                    lowpass_anchor = _avg("content_lowpass_anchor")
                    edge_anchor = _avg("content_edge_anchor")
                    if lowpass_anchor > 0.0 or edge_anchor > 0.0:
                        postfix["cla"] = f"{lowpass_anchor:.4f}"
                        postfix["cea"] = f"{edge_anchor:.4f}"
                progress.set_postfix(**postfix)

            data_wait_start = time.perf_counter()

            clear_model = getattr(self.model, "clear_runtime_caches", None)
            if callable(clear_model):
                clear_model()
            if self.teacher_model is not None:
                clear_teacher = getattr(self.teacher_model, "clear_runtime_caches", None)
                if callable(clear_teacher):
                    clear_teacher()

            del loss
            del loss_dict
            del batch

            if self.stop_after_global_steps > 0 and self.global_step >= self.stop_after_global_steps:
                self.requested_stop = True
                logger.info(
                    "Reached stop_after_global_steps=%d at epoch=%d batch=%d.",
                    self.stop_after_global_steps,
                    epoch,
                    step_idx,
                )
                break

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
        metrics.setdefault("curvature", 0.0)
        metrics.setdefault("distill_velocity", 0.0)
        metrics.setdefault("distill_endpoint", 0.0)
        metrics.setdefault("ot_cost", 0.0)
        metrics.setdefault("terminal_swd", 0.0)
        metrics.setdefault("terminal_swd_aux", 0.0)
        metrics.setdefault("cycle_consistency", 0.0)
        metrics.setdefault("content_lowpass_anchor", 0.0)
        metrics.setdefault("content_edge_anchor", 0.0)
        metrics.setdefault("aux_target_ratio", 0.0)
        metrics.setdefault("plan_entropy", 0.0)
        metrics.setdefault("semantic_attn_mean", 0.0)
        metrics.setdefault("semantic_k_abs", 0.0)
        metrics.setdefault("bridge_sigma", 0.0)
        metrics.setdefault("bridge_noise_schedule_exact", 0.0)
        metrics.setdefault("semantic_topology_attn_entropy", 0.0)
        metrics.setdefault("semantic_topology_attn_active", 0.0)
        metrics.setdefault("identity_ratio", 0.0)
        metrics.setdefault("t_mean", 0.0)
        metrics.setdefault("velocity_abs", 0.0)
        metrics.setdefault("endpoint_abs", 0.0)
        metrics.setdefault("velocity_max", 0.0)
        metrics.setdefault("endpoint_max", 0.0)
        metrics.setdefault("base_endpoint_abs", 0.0)
        metrics.setdefault("base_endpoint_max", 0.0)
        metrics.setdefault("final_endpoint_abs", 0.0)
        metrics.setdefault("final_endpoint_max", 0.0)
        metrics.setdefault("proximal_residual_abs", 0.0)
        metrics.setdefault("proximal_clamp_scale", 1.0)
        metrics.setdefault("proximal_residual_energy", 0.0)
        metrics.setdefault("base_transport_abs", 0.0)
        metrics.setdefault("proximal_to_transport_ratio", 0.0)
        metrics.setdefault("proximal_trust_penalty", 0.0)
        metrics.setdefault("teacher_alignment", 0.0)
        metrics.setdefault("teacher_abs", 0.0)
        metrics.setdefault("barycentric_entropy", 0.0)
        metrics.setdefault("kinetic_low_band", 0.0)
        metrics.setdefault("kinetic_high_band", 0.0)
        metrics["lr"] = float(self.optimizer.param_groups[0]["lr"])
        metrics["data_time_sec"] = data_time_total
        metrics["forward_time_sec"] = forward_time_total
        metrics["backward_time_sec"] = backward_time_total
        metrics["optimizer_time_sec"] = optimizer_time_total
        metrics["compute_time_sec"] = compute_time_total
        metrics["epoch_time_sec"] = epoch_time
        metrics["samples_seen"] = float(samples_seen)
        metrics["samples_per_sec"] = samples_per_sec
        if self.device.type == "cuda":
            metrics["cuda_peak_allocated_gb"] = float(torch.cuda.max_memory_allocated(self.device) / (1024**3))
            metrics["cuda_peak_reserved_gb"] = float(torch.cuda.max_memory_reserved(self.device) / (1024**3))
        else:
            metrics["cuda_peak_allocated_gb"] = 0.0
            metrics["cuda_peak_reserved_gb"] = 0.0
        return metrics

    def log_epoch(self, epoch: int, metrics: Dict[str, float]) -> None:
        append_training_log(self.log_file, metrics, epoch)

    def save_checkpoint(self, epoch: int, metrics: Dict[str, float], *, tag: str | None = None) -> Path:
        self._prune_checkpoint_threads()
        path = self.checkpoint_dir / ((tag or f"epoch_{epoch:04d}") + ".pt")
        model_for_state = unwrap_compiled_model(self.model)
        payload = {
            "epoch": int(epoch),
            "global_step": int(self.global_step),
            "model_state_dict": model_for_state.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict() if self.scheduler is not None else None,
            "loss_state_dict": self.loss_fn.state_dict() if hasattr(self.loss_fn, "state_dict") else None,
            "config": self.serialized_config,
            "metrics": metrics,
        }
        if self.async_checkpoint_save:
            payload = self._snapshot_for_checkpoint(payload)
            thread = threading.Thread(target=torch.save, args=(payload, path), daemon=False)
            thread.start()
            self._checkpoint_threads.append(thread)
            logger.info("Scheduled async checkpoint save: %s", path)
        else:
            torch.save(payload, path)
            logger.info("Saved checkpoint: %s", path)
        return path

    def maybe_save_step_checkpoint(self, epoch: int, step_idx: int, metrics: Dict[str, float]) -> Optional[Path]:
        if not self.save_step_milestones:
            return None
        if self.global_step not in self.save_step_milestones:
            return None
        if self.global_step in self._saved_step_milestones:
            return None
        payload = dict(metrics)
        payload["checkpoint_epoch"] = float(epoch)
        payload["checkpoint_step_in_epoch"] = float(step_idx)
        path = self.save_checkpoint(epoch, payload, tag=f"step_{self.global_step:06d}")
        self._saved_step_milestones.add(self.global_step)
        logger.info("Saved milestone checkpoint at global_step=%d -> %s", self.global_step, path)
        return path
