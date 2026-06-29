"""
Inference utilities for Latent AdaCUT.

Compatibility note:
This file keeps the historical `LGTInference` API so existing evaluation scripts
(`run_evaluation.py`) can be reused directly.
"""

from __future__ import annotations

import logging
import os
import sys
import json
import hashlib
from dataclasses import fields
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from PIL import Image

_SRC_ROOT = Path(__file__).resolve().parents[1]
if str(_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(_SRC_ROOT))

from config_schema import ExperimentConfig, load_config, merge_config_dicts, resolve_inference_section
from model import build_model_from_config
from style_families import prune_state_dict_for_tokenizer_family, validate_i2sb_contract, validate_pure_latent_contract
from utils.training import strip_compile_prefix

logger = logging.getLogger(__name__)

# Optional ModelScope support.
try:
    from modelscope.hub import snapshot_download as ms_snapshot_download  # type: ignore

    MODELSCOPE_AVAILABLE = True
except Exception:
    try:
        import modelscope.hub as ms_hub  # type: ignore

        ms_snapshot_download = getattr(ms_hub, "snapshot_download", ms_hub)
        MODELSCOPE_AVAILABLE = True
    except Exception:
        ms_snapshot_download = None
        MODELSCOPE_AVAILABLE = False


class VAEDecodeWrapper(torch.nn.Module):
    """Minimal tensor-only VAE decoder wrapper for torch.compile."""

    def __init__(self, vae: torch.nn.Module) -> None:
        super().__init__()
        self.post_quant_conv = vae.post_quant_conv
        self.decoder = vae.decoder

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        z = self.post_quant_conv(z)
        return self.decoder(z)


def configure_torch_compile_cache(cache_dir: str | os.PathLike | None) -> None:
    if cache_dir is None or not str(cache_dir).strip():
        return
    root = os.path.abspath(os.path.expanduser(str(cache_dir)))
    os.makedirs(root, exist_ok=True)
    inductor_dir = os.path.join(root, "inductor")
    triton_dir = os.path.join(root, "triton")
    os.makedirs(inductor_dir, exist_ok=True)
    os.makedirs(triton_dir, exist_ok=True)
    os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR", inductor_dir)
    os.environ.setdefault("TRITON_CACHE_DIR", triton_dir)


def _host_can_resolve_path(path: Path) -> bool:
    text = str(path)
    if os.name == "nt" and (
        text.startswith("/mnt/")
        or text.startswith("\\mnt\\")
        or text.startswith("/mnt\\")
        or text.startswith("\\mnt/")
    ):
        return False
    return True


def _resolve_optional_host_path(raw_path: str, *, base_dirs: list[Path]) -> Optional[Path]:
    text = str(raw_path or "").strip()
    if not text:
        return None
    candidate = Path(text)
    candidates: list[Path] = []
    if candidate.is_absolute():
        candidates.append(candidate)
    else:
        for base in base_dirs:
            candidates.append(base / candidate)
        candidates.append(Path.cwd() / candidate)
    seen: set[str] = set()
    for item in candidates:
        try:
            resolved = item.expanduser().resolve(strict=False)
        except Exception:
            resolved = item.expanduser()
        key = str(resolved)
        if key in seen:
            continue
        seen.add(key)
        if not _host_can_resolve_path(resolved):
            continue
        if resolved.exists():
            return resolved
    return None


class ORTVAEDecoder:
    """Fixed-shape ONNX Runtime VAE decoder with CUDA I/O binding."""

    def __init__(
        self,
        onnx_path: str | os.PathLike,
        *,
        device_id: int = 0,
        use_tensorrt: bool = False,
        trt_cache_dir: str | os.PathLike | None = None,
    ) -> None:
        import onnxruntime as ort

        self.onnx_path = str(Path(onnx_path).resolve())
        self.device_id = int(device_id)
        providers = []
        if use_tensorrt:
            cache_dir = str(Path(trt_cache_dir or Path(self.onnx_path).with_suffix(".trt_cache")).resolve())
            os.makedirs(cache_dir, exist_ok=True)
            providers.append(
                (
                    "TensorrtExecutionProvider",
                    {
                        "device_id": self.device_id,
                        "trt_engine_cache_enable": True,
                        "trt_engine_cache_path": cache_dir,
                        "trt_fp16_enable": True,
                    },
                )
            )
        providers.extend(
            [
                ("CUDAExecutionProvider", {"device_id": self.device_id}),
                "CPUExecutionProvider",
            ]
        )
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.session = ort.InferenceSession(self.onnx_path, sess_options=sess_options, providers=providers)
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name
        self.providers = self.session.get_providers()
        input_shape = list(self.session.get_inputs()[0].shape)
        self.fixed_batch = int(input_shape[0]) if input_shape and isinstance(input_shape[0], int) else None

    @torch.no_grad()
    def decode(self, latent: torch.Tensor, *, scaling_factor: float) -> torch.Tensor:
        latent = latent.to(device=f"cuda:{self.device_id}", dtype=torch.float16).contiguous()
        latent = (latent / max(float(scaling_factor), 1e-8)).contiguous()
        b, _, h, w = latent.shape
        run_latent = latent
        if self.fixed_batch is not None and b != self.fixed_batch:
            if b > self.fixed_batch:
                raise ValueError(
                    f"ORT VAE decoder was exported for batch={self.fixed_batch}, got batch={b}. "
                    "Lower --vae_decode_batch_size or export a matching decoder."
                )
            pad = torch.zeros(
                (self.fixed_batch - b, latent.shape[1], h, w),
                device=latent.device,
                dtype=latent.dtype,
            )
            run_latent = torch.cat([latent, pad], dim=0).contiguous()
        output = torch.empty((b, 3, h * 8, w * 8), device=latent.device, dtype=torch.float16).contiguous()
        run_output = output
        if run_latent.shape[0] != b:
            run_output = torch.empty(
                (run_latent.shape[0], 3, h * 8, w * 8),
                device=latent.device,
                dtype=torch.float16,
            ).contiguous()
        binding = self.session.io_binding()
        binding.bind_input(
            name=self.input_name,
            device_type="cuda",
            device_id=self.device_id,
            element_type=np.float16,
            shape=tuple(run_latent.shape),
            buffer_ptr=run_latent.data_ptr(),
        )
        binding.bind_output(
            name=self.output_name,
            device_type="cuda",
            device_id=self.device_id,
            element_type=np.float16,
            shape=tuple(run_output.shape),
            buffer_ptr=run_output.data_ptr(),
        )
        self.session.run_with_iobinding(binding)
        if run_output.shape[0] != b:
            output = run_output[:b]
        image = (output + 1.0) / 2.0
        return torch.clamp(image, 0.0, 1.0)


def _call_modelscope_snapshot(repo_id: str, dest: str):
    if not MODELSCOPE_AVAILABLE or ms_snapshot_download is None:
        raise RuntimeError("ModelScope snapshot downloader not available")

    if callable(ms_snapshot_download):
        last_exc = None
        for attempt in (
            lambda: ms_snapshot_download(repo_id, cache_dir=dest),
            lambda: ms_snapshot_download(repo_id, dest),
            lambda: ms_snapshot_download(repo_id=repo_id, cache_dir=dest),
        ):
            try:
                return attempt()
            except TypeError as e:
                last_exc = e
        raise last_exc or RuntimeError("Callable ms_snapshot_download failed")

    func = getattr(ms_snapshot_download, "snapshot_download", None) or getattr(
        ms_snapshot_download, "download", None
    )
    if callable(func):
        return func(repo_id, cache_dir=dest)
    raise RuntimeError("No callable snapshot_download available in ModelScope")


def _find_hf_repo_root(dest: str) -> Optional[str]:
    if not os.path.exists(dest):
        return None
    for root, _, files in os.walk(dest):
        if "config.json" in files or "model_index.json" in files or "pytorch_model.bin" in files:
            return root
    return None


class LGTInference:
    """
    Backward-compatible inference class for evaluation scripts.
    """

    def __init__(
        self,
        model_path,
        device="cuda",
        num_steps=1,
        step_size=None,
        style_strength=None,
        style_adapter_path=None,
        residual_scale=1.0,
        config_override_path=None,
    ):
        self.device = device
        self.num_steps = int(num_steps)
        self._style_adapter_path = style_adapter_path

        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        raw_config = checkpoint.get("config", {}) or {}
        if config_override_path:
            raw_config = merge_config_dicts(raw_config, load_config(config_override_path))
        self._runtime_config_signature = self._config_signature(raw_config)
        config = ExperimentConfig.from_mapping(raw_config)
        if isinstance(raw_config, dict):
            for section_name in ("model", "bridge", "training", "data", "checkpoint"):
                raw_section = raw_config.get(section_name, {})
                section_obj = getattr(config, section_name, None)
                if not isinstance(raw_section, dict) or section_obj is None:
                    continue
                for key, value in raw_section.items():
                    if hasattr(section_obj, key):
                        setattr(section_obj, key, value)
        bridge_cfg = config.bridge
        infer_cfg = resolve_inference_section(config)
        contract_family = str(getattr(config.model, "contract_family", "legacy")).strip().lower()
        self.objective_mode = str(bridge_cfg.objective_mode).strip().lower()
        if self.objective_mode in {"i2sb", "i2sb_endpoint", "bridge_endpoint"}:
            self.objective_mode = "i2sb_endpoint"
        if contract_family not in ("620_spatial_bridge", "620_spectral_ode"):
            validate_i2sb_contract(
                solver_family=str(getattr(config.model, "solver_family", "euler_legacy")),
                transport_prediction_mode=str(getattr(config.model, "transport_prediction_mode", "velocity")),
                objective_mode=self.objective_mode,
                loss_type=str(getattr(config.bridge, "loss_type", "")),
                bridge_noise_schedule=str(getattr(config.bridge, "bridge_noise_schedule", "auto")),
            )
        validate_pure_latent_contract(
            tokenizer_family=str(getattr(config.model, "tokenizer_family", "legacy_factorized")),
            style_tokenizer=str(getattr(config.model, "style_tokenizer", "")),
            semantic_supervision_family=str(getattr(config.bridge, "semantic_supervision_family", "legacy_terminal_swd")),
            dino_masked_swd_weight=float(getattr(config.bridge, "dino_masked_swd_weight", 0.0)),
            tokenizer_content_adaptive=bool(getattr(config.model, "tokenizer_content_adaptive", False)),
        )
        state_dict = strip_compile_prefix(checkpoint["model_state_dict"])
        state_dict, removed_contract_keys = prune_state_dict_for_tokenizer_family(
            state_dict,
            tokenizer_family=str(getattr(config.model, "tokenizer_family", "legacy_factorized")),
            contract_family=str(getattr(config.model, "contract_family", "legacy")),
            style_injection_mode=str(getattr(config.model, "style_injection_mode", "none")),
            proximal_mode=str(getattr(config.model, "proximal_mode", "off")),
            style_delta_mode=str(getattr(config.model, "style_delta_mode", "none")),
            output_appearance_alignment_mode=str(getattr(config.model, "output_appearance_alignment_mode", "none")),
        )
        if removed_contract_keys:
            logger.info(
                "Pruned %d legacy contract keys while loading inference checkpoint for tokenizer_family=%s",
                len(removed_contract_keys),
                str(getattr(config.model, "tokenizer_family", "legacy_factorized")),
            )

        self.model = build_model_from_config(config.model, bridge_cfg=config.bridge, use_checkpointing=False).to(device)
        # Eval-only override paths may need solver attributes that older
        # checkpoints/backbone constructors never materialized as runtime attrs.
        for key in (
            "solver_rk_order",
            "solver_corrector_steps",
            "solver_corrector_step_size",
            "solver_corrector_mode",
            "solver_corrector_lowpass_kernel",
            "solver_corrector_clamp",
            "solver_tangent_projection_strength",
            "solver_stochastic_noise_scale",
            "solver_fiber_aligned",
        ):
            if hasattr(config.model, key):
                setattr(self.model, key, getattr(config.model, key))
        try:
            self.model.load_state_dict(state_dict, strict=True)
        except RuntimeError as exc:
            if os.environ.get("SB_DEBUG_INFERENCE_LOAD", "").strip():
                debug_keys = [
                    "style_tokenizer",
                    "tokenizer_projection_mode",
                    "style_injection_mode",
                    "style_injection_form",
                    "proximal_mode",
                ]
                debug_values = {key: getattr(config.model, key, None) for key in debug_keys}
                raw_model = raw_config.get("model", {}) if isinstance(raw_config, dict) else {}
                raw_debug_values = {key: raw_model.get(key) for key in debug_keys} if isinstance(raw_model, dict) else {}
                available = set(dict(self.model.named_parameters()).keys()) | set(dict(self.model.named_buffers()).keys())
                model_field_names = [item.name for item in fields(type(config.model))]
                logger.warning(
                    "Inference debug schema: module=%s file=%s fields_has=%s",
                    type(config.model).__module__,
                    sys.modules.get(type(config.model).__module__).__file__ if sys.modules.get(type(config.model).__module__) is not None else None,
                    {
                        "tokenizer_projection_mode": "tokenizer_projection_mode" in model_field_names,
                        "style_injection_mode": "style_injection_mode" in model_field_names,
                        "style_injection_form": "style_injection_form" in model_field_names,
                        "proximal_mode": "proximal_mode" in model_field_names,
                    },
                )
                logger.warning("Inference debug raw config: %s", raw_debug_values)
                logger.warning("Inference debug config: %s", debug_values)
                logger.warning(
                    "Inference debug presence: %s",
                    {
                        "style_tokenizer.concept_atoms": "style_tokenizer.concept_atoms" in available,
                        "style_tokenizer.atom_logits.weight": "style_tokenizer.atom_logits.weight" in available,
                        "style_tokenizer.field_gates": "style_tokenizer.field_gates" in available,
                        "style_tokenizer.identity.weight": "style_tokenizer.identity.weight" in available,
                        "body_style_spatial_proj.0.weight": "body_style_spatial_proj.0.weight" in available,
                        "decoder_style_spatial_proj.0.weight": "decoder_style_spatial_proj.0.weight" in available,
                        "proximal_attn_q.weight": "proximal_attn_q.weight" in available,
                    },
                )
            logger.warning("Checkpoint/model key mismatch, falling back to non-strict load: %s", exc)
            self.model.load_state_dict(state_dict, strict=False)
        if style_adapter_path:
            self._load_style_adapter(style_adapter_path)
        self._maybe_load_transport_style_stats_bank(
            config=config,
            model_path=model_path,
            config_override_path=config_override_path,
        )
        self.model.eval()

        cfg_step = float(infer_cfg.get("step_size", 1.0))
        self.step_size = float(step_size if step_size is not None else cfg_step)
        cfg_strength = infer_cfg.get("style_strength")
        if style_strength is None and cfg_strength is None:
            self.style_strength = None
        else:
            self.style_strength = float(style_strength if style_strength is not None else cfg_strength)
        self.residual_scale = max(0.0, float(residual_scale))

    @staticmethod
    def _config_signature(raw_config: dict) -> str:
        """Hash the architecture/solver contract that determines module layout."""

        if not isinstance(raw_config, dict):
            raw_config = {}
        payload = {
            "model": raw_config.get("model", {}) or {},
            "bridge": raw_config.get("bridge", {}) or {},
            "inference": raw_config.get("inference", {}) or {},
        }
        encoded = json.dumps(payload, sort_keys=True, default=str, separators=(",", ":")).encode("utf-8")
        return hashlib.sha1(encoded).hexdigest()

    def reload_checkpoint(self, model_path, *, config_override_path=None) -> None:
        """Reuse the constructed inference module for another checkpoint from the same run."""

        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        raw_config = checkpoint.get("config", {}) or {}
        if config_override_path:
            raw_config = merge_config_dicts(raw_config, load_config(config_override_path))
        signature = self._config_signature(raw_config)
        if signature != self._runtime_config_signature:
            raise RuntimeError("checkpoint architecture signature changed; cannot reuse LGTInference")
        config = ExperimentConfig.from_mapping(raw_config)
        state_dict = strip_compile_prefix(checkpoint["model_state_dict"])
        state_dict, removed_contract_keys = prune_state_dict_for_tokenizer_family(
            state_dict,
            tokenizer_family=str(getattr(config.model, "tokenizer_family", "legacy_factorized")),
            contract_family=str(getattr(config.model, "contract_family", "legacy")),
            style_injection_mode=str(getattr(config.model, "style_injection_mode", "none")),
            proximal_mode=str(getattr(config.model, "proximal_mode", "off")),
            style_delta_mode=str(getattr(config.model, "style_delta_mode", "none")),
            output_appearance_alignment_mode=str(getattr(config.model, "output_appearance_alignment_mode", "none")),
        )
        if removed_contract_keys:
            logger.info(
                "Pruned %d legacy contract keys while reloading inference checkpoint for tokenizer_family=%s",
                len(removed_contract_keys),
                str(getattr(config.model, "tokenizer_family", "legacy_factorized")),
            )
        try:
            self.model.load_state_dict(state_dict, strict=True)
        except RuntimeError as exc:
            logger.warning("Checkpoint/model key mismatch during reload, falling back to non-strict load: %s", exc)
            self.model.load_state_dict(state_dict, strict=False)
        if self._style_adapter_path:
            self._load_style_adapter(self._style_adapter_path)
        self._maybe_load_transport_style_stats_bank(
            config=config,
            model_path=model_path,
            config_override_path=config_override_path,
        )
        self.model.eval()

    def _maybe_load_transport_style_stats_bank(
        self,
        *,
        config: ExperimentConfig,
        model_path: str | os.PathLike,
        config_override_path: str | os.PathLike | None,
    ) -> None:
        loader = getattr(self.model, "load_transport_style_stats_bank", None)
        if not callable(loader):
            return
        raw_path = str(getattr(config.model, "transport_stats_bank_path", "") or "").strip()
        if not raw_path:
            return
        base_dirs = [Path(model_path).resolve().parent, _SRC_ROOT.parent]
        if config_override_path:
            base_dirs.insert(0, Path(config_override_path).resolve().parent)
        resolved = _resolve_optional_host_path(raw_path, base_dirs=base_dirs)
        if resolved is None:
            required = bool(getattr(config.model, "transport_stats_bank_required", False))
            message = f"transport stats bank not found/resolvable on this host: {raw_path}"
            if required:
                raise FileNotFoundError(message)
            logger.warning("%s; continuing without bank.", message)
            return
        payload = loader(resolved)
        logger.info("Loaded transport stats bank from %s: %s", resolved, payload)

    def _load_style_adapter(self, style_adapter_path) -> None:
        adapter_path = os.path.expanduser(str(style_adapter_path))
        adapter = torch.load(adapter_path, map_location=self.device, weights_only=False)
        if not isinstance(adapter, dict):
            raise ValueError(f"Unsupported style adapter format: {adapter_path}")
        pure_latent_family = str(getattr(self.model, "tokenizer_family", "legacy_factorized")).strip().lower() in {
            "pure_latent_spatial",
            "smoe_translator",
            "affine_connection_tokenizer",
        }
        with torch.no_grad():
            tokenizer_state = {
                key.removeprefix("style_tokenizer."): value
                for key, value in adapter.items()
                if key.startswith("style_tokenizer.")
            }
            tokenizer_module = getattr(self.model, "style_tokenizer", None)
            if tokenizer_state and pure_latent_family:
                logger.warning(
                    "Ignoring legacy style_tokenizer adapter payload because tokenizer_family=%s uses structured_style_tokenizer as the active path.",
                    str(getattr(self.model, "tokenizer_family", "legacy_factorized")),
                )
            elif tokenizer_state and tokenizer_module is not None:
                tokenizer_module.load_state_dict(tokenizer_state, strict=False)
        logger.info("Loaded style adapter: %s", adapter_path)

    @torch.no_grad()
    def inversion(self, x1):
        # AdaCUT is direct mapping; inversion is identity for compatibility.
        return x1.clone()

    @torch.no_grad()
    def generation(self, x0, target_style_id, num_steps=None):
        return self.generation_with_target_latent(
            x0,
            target_style_id,
            num_steps=num_steps,
            target_style_latent=None,
        )

    @torch.no_grad()
    def generation_with_target_latent(self, x0, target_style_id, num_steps=None, target_style_latent=None, source_style_latent=None):
        if num_steps is None:
            num_steps = self.num_steps
        b = x0.shape[0]
        if isinstance(target_style_id, int):
            target_style_id = torch.full((b,), target_style_id, dtype=torch.long, device=x0.device)
        if self.objective_mode == "omf":
            endpoint = self.model.endpoint_map(
                x0,
                style_id=target_style_id,
                step_size=self.step_size,
                style_strength=self.style_strength,
                target_style_latent=target_style_latent,
            )
            if abs(self.residual_scale - 1.0) > 1e-6:
                return x0 + (endpoint - x0) * self.residual_scale
            return endpoint
        integrate_kwargs = {
            "step_size": self.step_size,
            "style_strength": self.style_strength,
        }
        if isinstance(target_style_latent, dict):
            if target_style_latent.get("style_dino_patches") is not None:
                integrate_kwargs["style_dino_patches"] = target_style_latent.get("style_dino_patches")
            if target_style_latent.get("style_dino_cls") is not None:
                integrate_kwargs["style_dino_cls"] = target_style_latent.get("style_dino_cls")
            if target_style_latent.get("style_text_tokens") is not None:
                integrate_kwargs["style_text_tokens"] = target_style_latent.get("style_text_tokens")
            # FC-SB Phase 3 deepfix: 传递 style_latent_tensor 让 N1 endpoint AdaIN 块能执行
            _style_latent_tensor = target_style_latent.get("style_latent_tensor")
            if _style_latent_tensor is not None:
                integrate_kwargs["target_style_latent"] = _style_latent_tensor
        else:
            integrate_kwargs["target_style_latent"] = target_style_latent
        # FC-SB Phase 4 A2 Step2: 传递 source_style_latent 让 fiber 空间 source-repulsion 能执行
        if source_style_latent is not None:
            if isinstance(source_style_latent, dict):
                _src_tensor = source_style_latent.get("style_latent_tensor")
                if _src_tensor is not None:
                    integrate_kwargs["source_style_latent"] = _src_tensor
            else:
                integrate_kwargs["source_style_latent"] = source_style_latent
        return self.model.integrate(
            x0,
            style_id=target_style_id,
            num_steps=max(1, int(num_steps)),
            **integrate_kwargs,
        )

    @torch.no_grad()
    def transfer_style(
        self,
        x_source,
        target_style_id,
        num_steps=None,
        return_intermediate=False,
        target_style_latent=None,
    ):
        x0 = self.inversion(x_source)
        x_target = self.generation_with_target_latent(
            x0,
            target_style_id,
            num_steps,
            target_style_latent=target_style_latent,
        )
        if return_intermediate:
            return x_target, x0
        return x_target

    @torch.no_grad()
    def interpolate_styles(self, x_source, style_ids, num_steps=None):
        if num_steps is None:
            num_steps = self.num_steps
        x0 = self.inversion(x_source)
        return [self.generation(x0, sid, num_steps) for sid in style_ids]


def download_vae_with_fallback(model_id, device="cuda", cache_dir=None):
    try:
        # Prefer the legacy direct module path first. Some newer diffusers
        # package-level imports eagerly import optional autoencoder families
        # that require newer transformers than our remote training env uses.
        from diffusers.models.autoencoder_kl import AutoencoderKL
    except Exception:
        from diffusers import AutoencoderKL

    force_dtype = torch.float16
    model_key = str(model_id).strip().lower()
    if model_key in {"sdxl-fp32", "sdxl-float32"}:
        model_id = "stabilityai/sdxl-vae"
        force_dtype = torch.float32
    elif model_key in {"sdxl-fp16-fix", "sdxl-fix"}:
        model_id = "madebyollin/sdxl-vae-fp16-fix"

    vae_presets = {
        "sd15": "stabilityai/sd-vae-ft-mse",
        "sdxl": "stabilityai/sdxl-vae",
        "mse": "stabilityai/sd-vae-ft-mse",
        "ema": "stabilityai/sd-vae-ft-ema",
    }
    preset = vae_presets.get(str(model_id).strip().lower())
    if preset:
        model_id = preset

    if cache_dir is None:
        cache_dir = os.path.expanduser("~/.cache/huggingface/hub")
    os.makedirs(cache_dir, exist_ok=True)

    try:
        vae = AutoencoderKL.from_pretrained(
            model_id,
            torch_dtype=force_dtype,
            cache_dir=cache_dir,
            local_files_only=True,
        ).to(device)
        vae.eval()
        return vae
    except Exception:
        pass

    ms_dest = os.path.join(cache_dir, "modelscope", model_id.replace("/", "_"))
    if os.path.exists(ms_dest):
        found = _find_hf_repo_root(ms_dest)
        if found:
            try:
                vae = AutoencoderKL.from_pretrained(found, torch_dtype=force_dtype, local_files_only=True).to(
                    device
                )
                vae.eval()
                return vae
            except Exception:
                pass

    if MODELSCOPE_AVAILABLE:
        try:
            dest = os.path.join(cache_dir, "modelscope", model_id.replace("/", "_"))
            os.makedirs(dest, exist_ok=True)
            ret = _call_modelscope_snapshot(model_id, dest)
            if isinstance(ret, str) and os.path.exists(ret):
                root = ret
            else:
                root = _find_hf_repo_root(dest)
            if root:
                vae = AutoencoderKL.from_pretrained(root, torch_dtype=force_dtype).to(device)
                vae.eval()
                return vae
        except Exception as exc:
            logger.warning("ModelScope VAE load failed: %s", exc)

    vae = AutoencoderKL.from_pretrained(model_id, torch_dtype=force_dtype, cache_dir=cache_dir).to(device)
    vae.eval()
    return vae


def load_vae(
    device="cuda",
    model_id="ema",
    cache_dir=None,
    *,
    enable_xformers: bool = True,
    compile_decoder: bool = False,
    compile_method: str = "pt2",
    compile_mode: str = "reduce-overhead",
    compile_fullgraph: bool = False,
    compile_cache_dir: str | os.PathLike | None = None,
):
    if device == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA not available, fallback to CPU for VAE.")
        device = "cpu"
    requested_compile_method = str(compile_method or "pt2").strip().lower()
    vae = download_vae_with_fallback(model_id, device=device, cache_dir=cache_dir)
    if str(device).startswith("cuda"):
        try:
            vae = vae.to(device=device, dtype=torch.float16, memory_format=torch.channels_last)
        except Exception:
            logger.debug("VAE channels_last conversion skipped.", exc_info=True)
        for method_name in ("disable_slicing", "disable_tiling"):
            try:
                method = getattr(vae, method_name, None)
                if callable(method):
                    method()
            except Exception:
                logger.debug("VAE %s skipped.", method_name, exc_info=True)
        try:
            if enable_xformers and requested_compile_method != "jit" and hasattr(vae, "enable_xformers_memory_efficient_attention"):
                vae.enable_xformers_memory_efficient_attention()
        except Exception:
            logger.debug("VAE xFormers attention not available; continue without it.", exc_info=True)
        if bool(compile_decoder):
            try:
                wrapper = VAEDecodeWrapper(vae).to(device=device, dtype=torch.float16, memory_format=torch.channels_last)
                wrapper.eval()
                method = requested_compile_method
                if method == "jit":
                    dummy_z = torch.randn(1, 4, 64, 64, device=device, dtype=torch.float16)
                    if str(device).startswith("cuda"):
                        dummy_z = dummy_z.contiguous(memory_format=torch.channels_last)
                    with torch.inference_mode():
                        vae.compiled_decoder = torch.jit.trace(wrapper, dummy_z, strict=False)
                        vae.compiled_decoder = torch.jit.freeze(vae.compiled_decoder.eval())
                    logger.info("Enabled TorchScript VAE decoder.")
                elif method in {"pt2", "compile", "torch_compile"}:
                    configure_torch_compile_cache(compile_cache_dir)
                    vae.compiled_decoder = torch.compile(
                        wrapper,
                        mode=str(compile_mode or "reduce-overhead"),
                        fullgraph=bool(compile_fullgraph),
                    )
                    logger.info(
                        "Enabled torch.compile VAE decoder: mode=%s fullgraph=%s",
                        str(compile_mode or "reduce-overhead"),
                        bool(compile_fullgraph),
                    )
                else:
                    raise ValueError(f"Unsupported VAE compile method: {compile_method}")
            except Exception:
                logger.exception("VAE decoder compile setup failed; falling back to diffusers decode.")
    return vae


@torch.no_grad()
def encode_image(vae, image_tensor, device="cuda"):
    image_tensor = image_tensor.to(device, dtype=torch.float16)
    if image_tensor.ndim == 4 and str(device).startswith("cuda"):
        image_tensor = image_tensor.contiguous(memory_format=torch.channels_last)
    latent = vae.encode(image_tensor).latent_dist.sample()
    latent = latent * vae.config.scaling_factor
    return latent


@torch.no_grad()
def decode_latent(vae, latent, device="cuda", scaling_factor=None):
    latent = latent.to(device, dtype=torch.float16)
    if latent.ndim == 4 and str(device).startswith("cuda"):
        latent = latent.contiguous(memory_format=torch.channels_last)
    scale = float(vae.config.scaling_factor if scaling_factor is None else scaling_factor)
    latent = latent / max(scale, 1e-8)
    compiled_decoder = getattr(vae, "compiled_decoder", None)
    if compiled_decoder is not None:
        try:
            image = compiled_decoder(latent)
        except Exception:
            logger.exception("Compiled VAE decoder failed; falling back to diffusers decode.")
            try:
                delattr(vae, "compiled_decoder")
            except Exception:
                pass
            image = vae.decode(latent).sample
    else:
        image = vae.decode(latent).sample
    image = (image + 1.0) / 2.0
    return torch.clamp(image, 0.0, 1.0)


def tensor_to_pil(tensor):
    if tensor.ndim == 4:
        tensor = tensor.squeeze(0)
    tensor = tensor.cpu().float()
    tensor = (tensor * 255).clamp(0, 255).to(torch.uint8)
    array = tensor.permute(1, 2, 0).numpy()
    return Image.fromarray(array)


if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("Usage: python utils/inference.py <checkpoint> <source_img> <output_path> [target_style_id] [style_adapter_path]")
        raise SystemExit(1)

    checkpoint_path = sys.argv[1]
    source_image_path = sys.argv[2]
    output_path = sys.argv[3]
    target_style_id = int(sys.argv[4]) if len(sys.argv) > 4 else 1
    style_adapter_path = sys.argv[5] if len(sys.argv) > 5 else None

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vae = load_vae(device=str(device))
    inf = LGTInference(checkpoint_path, device=str(device), num_steps=1, style_adapter_path=style_adapter_path)
    model_scale = float(getattr(inf.model, "latent_scale_factor", 0.18215))
    vae_scale = float(getattr(getattr(vae, "config", None), "scaling_factor", model_scale))
    scale_in = model_scale / max(vae_scale, 1e-8)
    scale_out = vae_scale / max(model_scale, 1e-8)
    if abs(scale_in - 1.0) > 1e-4:
        print(f"WARNING: latent scale mismatch (model={model_scale:.6f}, vae={vae_scale:.6f}). Applying rescale.")

    image = Image.open(source_image_path).convert("RGB").resize((256, 256))
    image_tensor = torch.from_numpy(np.array(image)).float() / 255.0
    image_tensor = image_tensor.permute(2, 0, 1).unsqueeze(0)
    image_tensor = image_tensor * 2.0 - 1.0

    z = encode_image(vae, image_tensor, device=str(device))
    if abs(scale_in - 1.0) > 1e-4:
        z = z * scale_in
    z_out = inf.transfer_style(z, target_style_id=target_style_id, num_steps=1)
    if abs(scale_out - 1.0) > 1e-4:
        z_out = z_out * scale_out
    out = decode_latent(vae, z_out, device=str(device))
    tensor_to_pil(out).save(output_path)
    print(f"Saved: {output_path}")
