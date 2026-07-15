"""Internal gradient/information-flow probe for the current WEAVE baseline.

This probe does not use evaluation metrics. It runs training-style forward and
backward passes on the baseline checkpoint and records:

- module activation RMS and activation-gradient RMS;
- parameter gradient norms grouped by information path;
- style-memory/style-gate gradient strength;
- velocity sensitivity when only the learned style_id is swapped.

The output is meant to support mechanism diagnosis before trying changes.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from config_schema import load_experiment_config  # noqa: E402
from flow import FlowMatchingObjective  # noqa: E402
from model import build_model_from_config  # noqa: E402
from style_families import prune_state_dict_for_tokenizer_family  # noqa: E402
from utils.dataset import AdaCUTLatentDataset  # noqa: E402
from utils.training import strip_compile_prefix  # noqa: E402
from wavelet import dwt2_haar, idwt2_haar, subband_gamma_tensor  # noqa: E402

def _mean(values: list[float]) -> float:
    return float(sum(values) / max(1, len(values)))


def _rms(t: torch.Tensor) -> float:
    return float(t.detach().float().pow(2).mean().sqrt().cpu().item())


def _abs_mean(t: torch.Tensor) -> float:
    return float(t.detach().float().abs().mean().cpu().item())


def _to_float(value: Any) -> float:
    if torch.is_tensor(value):
        return float(value.detach().float().mean().cpu().item())
    return float(value)


def _grad_norm(parameters: list[torch.nn.Parameter]) -> float:
    total = 0.0
    for param in parameters:
        if param.grad is None:
            continue
        g = param.grad.detach().float()
        total += float(g.pow(2).sum().cpu().item())
    return math.sqrt(total)


def _param_norm(parameters: list[torch.nn.Parameter]) -> float:
    total = 0.0
    for param in parameters:
        p = param.detach().float()
        total += float(p.pow(2).sum().cpu().item())
    return math.sqrt(total)


def _first_existing_dir(candidates: list[Path], style_subdirs: list[str]) -> Path | None:
    for candidate in candidates:
        if candidate.exists() and all((candidate / subdir).exists() for subdir in style_subdirs):
            return candidate
    return None


def build_dataset(config: Any, batch_size: int, data_root: Path | None, latent_cache_mode: str) -> AdaCUTLatentDataset:
    data_cfg = config.data
    style_subdirs = list(data_cfg.style_subdirs)
    resolved_root = data_root
    if resolved_root is None:
        configured_root = Path(str(data_cfg.data_root))
        if configured_root.exists():
            resolved_root = configured_root
        else:
            resolved_root = _first_existing_dir(
                [
                    ROOT / "exp" / "dino_s_break" / "data" / "train",
                    ROOT / "exp" / "72_fewshot" / "data" / "5p1_shot50" / "train",
                    Path("G:/wikiart27_latents_compact/train"),
                    Path("Y:/Latent_Style/style_data/wikiart27_latents_compact/train"),
                ],
                style_subdirs,
            )
    if resolved_root is None:
        raise FileNotFoundError(
            "Could not find a latent data_root containing all configured style_subdirs. "
            "Pass --data-root explicitly."
        )
    cache_mode = str(latent_cache_mode).strip().lower()
    latent_cache_dir = ""
    if cache_mode != "off":
        latent_cache_dir = str(data_cfg.latent_cache_dir)
    dataset = AdaCUTLatentDataset(
        data_root=str(resolved_root),
        style_subdirs=style_subdirs,
        allow_hflip=False,
        identity_ratio=None,
        batch_size_hint=batch_size,
        balance_target_styles_per_batch=False,
        preload_to_gpu=False,
        preload_max_vram_gb=0.0,
        preload_reserve_ratio=0.35,
        virtual_length_multiplier=1.0,
        content_style_sampling_weights=None,
        target_style_sampling_weights=None,
        pairing_cache_path=data_cfg.pairing_cache_path,
        pairing_cache_topk=int(data_cfg.pairing_cache_topk),
        pairing_cache_active_topk=int(data_cfg.pairing_cache_active_topk),
        pairing_cache_sample_mode=str(data_cfg.pairing_cache_sample_mode),
        pairing_cache_rank_schedule=str(data_cfg.pairing_cache_rank_schedule),
        pairing_cache_min_topk=int(data_cfg.pairing_cache_min_topk),
        pairing_cache_curriculum_epochs=0,
        pairing_cache_rank_power=float(data_cfg.pairing_cache_rank_power),
        pairing_cache_explore_prob=0.0,
        pairing_cache_explore_topk=0,
        pairing_cache_dual_target_mix=0.0,
        pairing_cache_dual_target_topk=0,
        pairing_cache_aux_target_topk=0,
        pairing_cache_cross_only=bool(data_cfg.pairing_cache_cross_only),
        latent_cache_mode=cache_mode,
        latent_cache_dir=latent_cache_dir,
        style_caption_path="",
        device="cpu",
    )
    dataset._probe_data_root = str(resolved_root)  # type: ignore[attr-defined]
    dataset._probe_latent_cache_mode = cache_mode  # type: ignore[attr-defined]
    if hasattr(dataset, "set_epoch"):
        dataset.set_epoch(0)
    return dataset


def build_and_load_model(config: Any, checkpoint: Path, device: torch.device) -> torch.nn.Module:
    model = build_model_from_config(config.model, bridge_cfg=config.bridge).to(device)
    ckpt = torch.load(str(checkpoint), map_location="cpu", weights_only=False)
    state = ckpt.get("model_state_dict", ckpt.get("state_dict", ckpt))
    state = strip_compile_prefix(state)
    state, _ = prune_state_dict_for_tokenizer_family(
        state,
        tokenizer_family=str(getattr(config.model, "tokenizer_family", "legacy_factorized")),
        contract_family=str(getattr(config.model, "contract_family", "legacy")),
        style_injection_mode=str(getattr(config.model, "style_injection_mode", "none")),
        proximal_mode=str(getattr(config.model, "proximal_mode", "off")),
        style_delta_mode=str(getattr(config.model, "style_delta_mode", "none")),
        output_appearance_alignment_mode=str(getattr(config.model, "output_appearance_alignment_mode", "none")),
    )
    missing, unexpected = model.load_state_dict(state, strict=False)
    model._probe_load_info = {"missing": len(missing), "unexpected": len(unexpected)}  # type: ignore[attr-defined]
    return model


def move_batch(raw: dict[str, Any], device: torch.device) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in raw.items():
        if torch.is_tensor(value):
            if key in {"target_style_id", "source_style_id"}:
                out[key] = value.to(device=device, dtype=torch.long)
            else:
                out[key] = value.to(device=device, dtype=torch.float32)
        else:
            out[key] = value
    return out


class ActivationProbe:
    def __init__(self) -> None:
        self.rows: dict[str, list[dict[str, float]]] = defaultdict(list)
        self.handles: list[Any] = []

    def add(self, name: str, module: torch.nn.Module) -> None:
        def hook(_module: torch.nn.Module, _inputs: tuple[Any, ...], output: Any) -> None:
            tensor = output
            if isinstance(output, dict):
                return
            if isinstance(output, (tuple, list)):
                tensor = next((x for x in output if torch.is_tensor(x)), None)
            if not torch.is_tensor(tensor):
                return
            row: dict[str, float] = {
                "act_rms": _rms(tensor),
                "act_abs": _abs_mean(tensor),
            }

            def grad_hook(grad: torch.Tensor) -> None:
                row["grad_rms"] = _rms(grad)
                row["grad_abs"] = _abs_mean(grad)

            if tensor.requires_grad:
                tensor.register_hook(grad_hook)
            self.rows[name].append(row)

        self.handles.append(module.register_forward_hook(hook))

    def close(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles.clear()

    def summary(self) -> dict[str, dict[str, float]]:
        out: dict[str, dict[str, float]] = {}
        for name, rows in self.rows.items():
            keys = sorted({key for row in rows for key in row})
            out[name] = {key: _mean([row[key] for row in rows if key in row]) for key in keys}
        return out


def attach_activation_probe(model: torch.nn.Module) -> ActivationProbe:
    probe = ActivationProbe()
    probe.add("style_conditioner.patch_proj", model.style_conditioner.patch_proj)
    if getattr(model, "target_latent_tokenizer", None) is not None:
        probe.add("target_latent_tokenizer", model.target_latent_tokenizer)
    if getattr(model, "target_latent_token_proj", None) is not None:
        probe.add("target_latent_token_proj", model.target_latent_token_proj)
    for name in (
        "target_latent_hf_encoder",
        "target_latent_hf_proj",
        "target_latent_hf_spatial_lh",
        "target_latent_hf_spatial_hl",
        "target_latent_hf_spatial_hh",
        "target_latent_hf_subband_encoder_lh",
        "target_latent_hf_subband_encoder_hl",
        "target_latent_hf_subband_encoder_hh",
        "target_latent_hf_subband_proj_lh",
        "target_latent_hf_subband_proj_hl",
        "target_latent_hf_subband_proj_hh",
        "target_latent_hf_subband_delta_lh",
        "target_latent_hf_subband_delta_hl",
        "target_latent_hf_subband_delta_hh",
        "target_latent_hf_texture_encoder_lh",
        "target_latent_hf_texture_encoder_hl",
        "target_latent_hf_texture_encoder_hh",
        "target_latent_hf_texture_delta_lh",
        "target_latent_hf_texture_delta_hl",
        "target_latent_hf_texture_delta_hh",
    ):
        module = getattr(model, name, None)
        if module is not None:
            probe.add(name, module)
    probe.add("input_proj", model.input_proj)
    probe.add("time_proj", model.time_proj)
    for idx, block in enumerate(getattr(model, "blocks", [])):
        probe.add(f"block{idx}.residual", block)
        probe.add(f"block{idx}.sa_qkv", block.sa_qkv)
        probe.add(f"block{idx}.ca_q", block.q_proj)
        probe.add(f"block{idx}.ca_k", block.k_proj)
        probe.add(f"block{idx}.ca_v", block.v_proj)
        probe.add(f"block{idx}.ca_out", block.out_proj)
        probe.add(f"block{idx}.ffn", block.ffn)
    probe.add("head_ll", model.head_ll)
    probe.add("head_lh", model.head_lh)
    probe.add("head_hl", model.head_hl)
    if getattr(model, "head_hh", None) is not None:
        probe.add("head_hh", model.head_hh)
    return probe


def _module_params(module: torch.nn.Module | None) -> list[torch.nn.Parameter]:
    return list(module.parameters()) if module is not None else []


def _append_param(params: list[torch.nn.Parameter], param: Any) -> None:
    if isinstance(param, torch.nn.Parameter):
        params.append(param)


def module_groups(model: torch.nn.Module) -> dict[str, list[torch.nn.Parameter]]:
    groups: dict[str, list[torch.nn.Parameter]] = {
        "style_memory": [model.style_conditioner.style_memory],
        "style_conditioner.patch_proj": list(model.style_conditioner.patch_proj.parameters()),
        "input_proj": list(model.input_proj.parameters()),
        "time_proj": list(model.time_proj.parameters()),
        "head_ll": list(model.head_ll.parameters()),
        "head_lh": list(model.head_lh.parameters()),
        "head_hl": list(model.head_hl.parameters()),
    }
    if getattr(model, "head_hh", None) is not None:
        groups["head_hh"] = list(model.head_hh.parameters())
    if getattr(model, "target_latent_tokenizer", None) is not None:
        params = list(model.target_latent_tokenizer.parameters())
        params += list(model.target_latent_token_proj.parameters())
        if getattr(model, "target_latent_token_gate", None) is not None:
            params.append(model.target_latent_token_gate)
        groups["target_latent_token_fusion"] = params
    target_hf_head_params = (
        _module_params(getattr(model, "target_latent_hf_encoder", None))
        + _module_params(getattr(model, "target_latent_hf_proj", None))
        + _module_params(getattr(model, "target_latent_hf_delta_lh", None))
        + _module_params(getattr(model, "target_latent_hf_delta_hl", None))
        + _module_params(getattr(model, "target_latent_hf_delta_hh", None))
    )
    _append_param(target_hf_head_params, getattr(model, "target_latent_hf_gate", None))
    if target_hf_head_params:
        groups["target_hf_head_fusion"] = target_hf_head_params
    target_hf_spatial_params = (
        _module_params(getattr(model, "target_latent_hf_spatial_lh", None))
        + _module_params(getattr(model, "target_latent_hf_spatial_hl", None))
        + _module_params(getattr(model, "target_latent_hf_spatial_hh", None))
        + _module_params(getattr(model, "target_latent_hf_spatial_delta_lh", None))
        + _module_params(getattr(model, "target_latent_hf_spatial_delta_hl", None))
        + _module_params(getattr(model, "target_latent_hf_spatial_delta_hh", None))
    )
    if target_hf_spatial_params:
        groups["target_hf_spatial_fusion"] = target_hf_spatial_params
    target_hf_subband_params = (
        _module_params(getattr(model, "target_latent_hf_subband_encoder_lh", None))
        + _module_params(getattr(model, "target_latent_hf_subband_encoder_hl", None))
        + _module_params(getattr(model, "target_latent_hf_subband_encoder_hh", None))
        + _module_params(getattr(model, "target_latent_hf_subband_proj_lh", None))
        + _module_params(getattr(model, "target_latent_hf_subband_proj_hl", None))
        + _module_params(getattr(model, "target_latent_hf_subband_proj_hh", None))
        + _module_params(getattr(model, "target_latent_hf_subband_delta_lh", None))
        + _module_params(getattr(model, "target_latent_hf_subband_delta_hl", None))
        + _module_params(getattr(model, "target_latent_hf_subband_delta_hh", None))
    )
    _append_param(target_hf_subband_params, getattr(model, "target_latent_hf_subband_head_gate", None))
    if target_hf_subband_params:
        groups["target_hf_subband_fusion"] = target_hf_subband_params
    target_hf_texture_params = (
        _module_params(getattr(model, "target_latent_hf_texture_encoder_lh", None))
        + _module_params(getattr(model, "target_latent_hf_texture_encoder_hl", None))
        + _module_params(getattr(model, "target_latent_hf_texture_encoder_hh", None))
        + _module_params(getattr(model, "target_latent_hf_texture_delta_lh", None))
        + _module_params(getattr(model, "target_latent_hf_texture_delta_hl", None))
        + _module_params(getattr(model, "target_latent_hf_texture_delta_hh", None))
    )
    if target_hf_texture_params:
        groups["target_hf_texture_fusion"] = target_hf_texture_params
    for idx, block in enumerate(getattr(model, "blocks", [])):
        groups[f"block{idx}.adaln"] = list(
            getattr(block, "time_style_adaln", getattr(block, "time_adaln")).parameters()
        )
        groups[f"block{idx}.self_attn"] = list(block.sa_qkv.parameters()) + list(block.sa_out.parameters())
        groups[f"block{idx}.cross_attn_q"] = list(block.q_proj.parameters())
        groups[f"block{idx}.cross_attn_kv"] = list(block.k_proj.parameters()) + list(block.v_proj.parameters())
        groups[f"block{idx}.cross_attn_out_gate"] = list(block.out_proj.parameters()) + [block.style_gate]
        groups[f"block{idx}.ffn"] = list(block.ffn.parameters())
    return groups


def collect_grad_summary(model: torch.nn.Module) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    for name, params in module_groups(model).items():
        pnorm = _param_norm(params)
        gnorm = _grad_norm(params)
        out[name] = {
            "param_norm": pnorm,
            "grad_norm": gnorm,
            "grad_over_param": float(gnorm / (pnorm + 1e-12)),
        }
    return out


def collect_input_grad_summary(batch: dict[str, Any]) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    for key in ("content", "target_style"):
        value = batch.get(key)
        if not torch.is_tensor(value):
            continue
        grad = value.grad
        out[key] = {
            "tensor_rms": _rms(value),
            "grad_rms": _rms(grad) if grad is not None else 0.0,
            "grad_abs": _abs_mean(grad) if grad is not None else 0.0,
            "grad_over_tensor": (_rms(grad) / (_rms(value) + 1e-12)) if grad is not None else 0.0,
        }
    return out


def aggregate_path_summary(grad_summary: dict[str, dict[str, float]]) -> dict[str, dict[str, float]]:
    specs = {
        "style_memory": lambda name: name == "style_memory",
        "style_patch_proj": lambda name: name == "style_conditioner.patch_proj",
        "target_latent_fusion": lambda name: name == "target_latent_token_fusion",
        "target_hf_fusion": lambda name: name.startswith("target_hf_"),
        "target_hf_head": lambda name: name == "target_hf_head_fusion",
        "target_hf_spatial": lambda name: name == "target_hf_spatial_fusion",
        "target_hf_subband": lambda name: name == "target_hf_subband_fusion",
        "target_hf_texture": lambda name: name == "target_hf_texture_fusion",
        "input_time": lambda name: name in {"input_proj", "time_proj"},
        "self_attn": lambda name: ".self_attn" in name,
        "cross_attn_q": lambda name: ".cross_attn_q" in name,
        "cross_attn_kv": lambda name: ".cross_attn_kv" in name,
        "cross_attn_out_gate": lambda name: ".cross_attn_out_gate" in name,
        "adaln": lambda name: ".adaln" in name,
        "ffn": lambda name: ".ffn" in name,
        "head_ll": lambda name: name == "head_ll",
        "head_hf": lambda name: name in {"head_lh", "head_hl", "head_hh"},
    }
    out: dict[str, dict[str, float]] = {}
    for path_name, predicate in specs.items():
        rows = [row for name, row in grad_summary.items() if predicate(name)]
        if not rows:
            continue
        grad_norm = math.sqrt(sum(float(row["grad_norm"]) ** 2 for row in rows))
        param_norm = math.sqrt(sum(float(row["param_norm"]) ** 2 for row in rows))
        out[path_name] = {
            "grad_norm": grad_norm,
            "param_norm": param_norm,
            "grad_over_param": float(grad_norm / (param_norm + 1e-12)),
        }
    return out


def collect_loss_path_gradients(
    model: torch.nn.Module,
    loss_fn: FlowMatchingObjective,
    batch: dict[str, Any],
    config: Any,
) -> dict[str, dict[str, dict[str, float]]]:
    keys = [
        "loss",
        "loss_fm_hf_total",
        "loss_fm_spectral_ll",
        "loss_fm_spectral_lh",
        "loss_fm_spectral_hl",
        "loss_fm_spectral_hh",
        "loss_stat",
        "loss_stat_lh",
        "loss_stat_hl",
        "loss_stat_hh",
    ]
    content = batch["content"].detach()
    t = torch.full((content.shape[0],), 0.5, device=content.device, dtype=content.dtype)
    noise = torch.randn_like(content) if float(getattr(config.bridge, "bridge_sigma", 0.0)) > 0.0 else None
    out: dict[str, dict[str, dict[str, float]]] = {}
    for key in keys:
        model.zero_grad(set_to_none=True)
        detached = {
            name: value.detach()
            if torch.is_tensor(value) and value.is_floating_point()
            else value
            for name, value in batch.items()
        }
        metrics = spectral_losses_with_graph(model, loss_fn, detached, t=t, noise=noise)
        if key not in metrics:
            continue
        scalar = metrics[key]
        if scalar.requires_grad:
            scalar.backward()
        out[key] = aggregate_path_summary(collect_grad_summary(model))
    model.zero_grad(set_to_none=True)
    return out


def _flatten_current_grads(model: torch.nn.Module) -> torch.Tensor:
    chunks: list[torch.Tensor] = []
    for param in model.parameters():
        if not param.requires_grad:
            continue
        if param.grad is None:
            chunks.append(torch.zeros(param.numel(), device=param.device, dtype=torch.float32))
        else:
            chunks.append(param.grad.detach().float().reshape(-1))
    if not chunks:
        return torch.empty(0)
    return torch.cat(chunks)


def collect_loss_gradient_cosines(
    model: torch.nn.Module,
    loss_fn: FlowMatchingObjective,
    batch: dict[str, Any],
    config: Any,
) -> dict[str, float]:
    keys = [
        "loss_fm_hf_total",
        "loss_stat",
        "loss_fm_spectral_lh",
        "loss_stat_lh",
        "loss_fm_spectral_hl",
        "loss_stat_hl",
        "loss_fm_spectral_hh",
        "loss_stat_hh",
    ]
    content = batch["content"].detach()
    t = torch.full((content.shape[0],), 0.5, device=content.device, dtype=content.dtype)
    noise = torch.randn_like(content) if float(getattr(config.bridge, "bridge_sigma", 0.0)) > 0.0 else None
    vectors: dict[str, torch.Tensor] = {}
    for key in keys:
        model.zero_grad(set_to_none=True)
        detached = {
            name: value.detach()
            if torch.is_tensor(value) and value.is_floating_point()
            else value
            for name, value in batch.items()
        }
        metrics = spectral_losses_with_graph(model, loss_fn, detached, t=t, noise=noise)
        scalar = metrics.get(key)
        if scalar is None or not scalar.requires_grad:
            continue
        scalar.backward()
        vector = _flatten_current_grads(model)
        if vector.numel() > 0 and float(vector.norm().cpu().item()) > 0.0:
            vectors[key] = vector.detach().cpu()
    model.zero_grad(set_to_none=True)

    pairs = {
        "cos_fm_hf_vs_stat": ("loss_fm_hf_total", "loss_stat"),
        "cos_lh_mse_vs_stat": ("loss_fm_spectral_lh", "loss_stat_lh"),
        "cos_hl_mse_vs_stat": ("loss_fm_spectral_hl", "loss_stat_hl"),
        "cos_hh_mse_vs_stat": ("loss_fm_spectral_hh", "loss_stat_hh"),
    }
    out: dict[str, float] = {}
    for name, (left, right) in pairs.items():
        if left not in vectors or right not in vectors:
            continue
        lvec = vectors[left]
        rvec = vectors[right]
        denom = float(lvec.norm().item() * rvec.norm().item())
        out[name] = float(torch.dot(lvec, rvec).item() / (denom + 1e-12))
    for key, vector in vectors.items():
        out[f"grad_norm_{key}"] = float(vector.norm().item())
    return out


def spectral_losses_with_graph(
    model: torch.nn.Module,
    loss_fn: FlowMatchingObjective,
    batch: dict[str, Any],
    *,
    t: torch.Tensor,
    noise: torch.Tensor | None,
) -> dict[str, torch.Tensor]:
    content = batch["content"]
    target_style = batch["target_style"]
    target_style_id = batch["target_style_id"]
    style_text_tokens = batch.get("target_style_text_tokens")
    style_latent = batch.get("target_style_latent")
    if not torch.is_tensor(style_text_tokens):
        style_text_tokens = None
    if not torch.is_tensor(style_latent):
        style_latent = target_style

    target = target_style
    if loss_fn.latent_adain_enabled:
        content = loss_fn._adain_blend(content, target, loss_fn.latent_adain_gamma)

    if loss_fn.structure_aligned_target:
        ll_c, lh_c, hl_c, hh_c = dwt2_haar(content)
        ll_t, lh_t, hl_t, hh_t = dwt2_haar(target)
        if loss_fn.multi_level_dwt_enabled:
            ll2_c, lh2_c, hl2_c, hh2_c = dwt2_haar(ll_c)
            ll2_t, lh2_t, hl2_t, hh2_t = dwt2_haar(ll_t)
            del ll2_t
            a2 = loss_fn.multi_level_dwt_alpha2
            lh2_blend = (1.0 - a2) * lh2_c + a2 * lh2_t
            hl2_blend = (1.0 - a2) * hl2_c + a2 * hl2_t
            hh2_blend = (1.0 - a2) * hh2_c + a2 * hh2_t
            ll_c = idwt2_haar(ll2_c, lh2_blend, hl2_blend, hh2_blend)
        elif loss_fn.ll_partial_style_enabled and 0.0 < loss_fn.ll_partial_alpha <= 1.0:
            ll_c = loss_fn._partial_style_ll(ll_c, ll_t, loss_fn.ll_partial_alpha)
        if loss_fn.hf_wct_enabled:
            lh_t = loss_fn._wct_match_hf(lh_c, lh_t, loss_fn.hf_wct_beta)
            hl_t = loss_fn._wct_match_hf(hl_c, hl_t, loss_fn.hf_wct_beta)
            hh_t = loss_fn._wct_match_hf(hh_c, hh_t, loss_fn.hf_wct_beta)
        if loss_fn.hf_adain_enabled:
            lh_t = loss_fn._adain_blend(lh_c, lh_t, loss_fn.hf_adain_alpha_lh)
            hl_t = loss_fn._adain_blend(hl_c, hl_t, loss_fn.hf_adain_alpha_hl)
            hh_t = loss_fn._adain_blend(hh_c, hh_t, loss_fn.hf_adain_alpha_hh)
        if loss_fn.hf_overstylize_beta > 1.0:
            b = loss_fn.hf_overstylize_beta
            lh_t = (1.0 - b) * lh_c + b * lh_t
            hl_t = (1.0 - b) * hl_c + b * hl_t
            hh_t = (1.0 - b) * hh_c + b * hh_t
        target = idwt2_haar(ll_c, lh_t, hl_t, hh_t)

    if loss_fn.train_adain_enabled and loss_fn.train_adain_scale > 0.0 and torch.is_tensor(style_latent):
        target = loss_fn._apply_train_adain(target, style_latent)

    t_view = t.view(-1, 1, 1, 1).to(dtype=content.dtype)
    if loss_fn.bridge_sigma > 0.0:
        eps = torch.zeros_like(content) if noise is None else noise
        eps = eps * loss_fn.bridge_sigma
        if loss_fn.training_sde_noise_mode == "subtractive":
            x_t = (1.0 - t_view) * content + t_view * target - eps * (t_view * (1.0 - t_view)).sqrt()
        else:
            x_t = (1.0 - t_view) * content + t_view * target + eps * (t_view * (1.0 - t_view)).sqrt()
    else:
        x_t = (1.0 - t_view) * content + t_view * target

    target_delta = target - content
    target_ll, target_lh, target_hl, target_hh = dwt2_haar(target_delta)
    v_dict = model(
        x_t,
        t=t,
        style_id=target_style_id,
        style_latent=style_latent,
        style_text_tokens=style_text_tokens,
    )

    if loss_fn.subband_time_schedule_enabled:
        g_ll = subband_gamma_tensor(t, loss_fn.subband_gamma_ll).view(-1, 1, 1, 1).to(dtype=content.dtype)
        g_lh = subband_gamma_tensor(t, loss_fn.subband_gamma_lh).view(-1, 1, 1, 1).to(dtype=content.dtype)
        g_hl = subband_gamma_tensor(t, loss_fn.subband_gamma_hl).view(-1, 1, 1, 1).to(dtype=content.dtype)
        g_hh = subband_gamma_tensor(t, loss_fn.subband_gamma_hh).view(-1, 1, 1, 1).to(dtype=content.dtype)
        loss_ll = (g_ll * (v_dict["ll"].float() - target_ll.float()) ** 2).mean()
        loss_lh = (g_lh * (v_dict["lh"].float() - target_lh.float()) ** 2).mean()
        loss_hl = (g_hl * (v_dict["hl"].float() - target_hl.float()) ** 2).mean()
        loss_hh = content.new_tensor(0.0)
        if "hh" in v_dict:
            loss_hh = (g_hh * (v_dict["hh"].float() - target_hh.float()) ** 2).mean()
    else:
        loss_ll = loss_fn._fm_loss(v_dict["ll"], target_ll)
        loss_lh = loss_fn._fm_loss(v_dict["lh"], target_lh)
        loss_hl = loss_fn._fm_loss(v_dict["hl"], target_hl)
        loss_hh = content.new_tensor(0.0)
        if "hh" in v_dict:
            loss_hh = loss_fn._fm_loss(v_dict["hh"], target_hh)

    weighted_ll = loss_fn.w_ll * loss_ll
    weighted_lh = loss_fn.w_lh * loss_lh
    weighted_hl = loss_fn.w_hl * loss_hl
    weighted_hh = loss_fn.w_hh * loss_hh if "hh" in v_dict else content.new_tensor(0.0)
    stat_lh = content.new_tensor(0.0)
    stat_hl = content.new_tensor(0.0)
    stat_hh = content.new_tensor(0.0)
    if loss_fn.hf_stat_loss_enabled:
        stat_lh = loss_fn.hf_stat_weight * loss_fn._statistical_loss(v_dict["lh"], target_lh)
        stat_hl = loss_fn.hf_stat_weight * loss_fn._statistical_loss(v_dict["hl"], target_hl)
        if "hh" in v_dict:
            stat_hh = loss_fn.hf_stat_weight * loss_fn._statistical_loss(v_dict["hh"], target_hh)
    loss_fm = weighted_ll + weighted_lh + weighted_hl + weighted_hh
    loss_stat = stat_lh + stat_hl + stat_hh
    return {
        "loss": loss_fm + loss_stat,
        "loss_fm_hf_total": weighted_lh + weighted_hl + weighted_hh,
        "loss_fm_spectral_ll": weighted_ll,
        "loss_fm_spectral_lh": weighted_lh,
        "loss_fm_spectral_hl": weighted_hl,
        "loss_fm_spectral_hh": weighted_hh,
        "loss_stat": loss_stat,
        "loss_stat_lh": stat_lh,
        "loss_stat_hl": stat_hl,
        "loss_stat_hh": stat_hh,
    }


def collect_block_debug(model: torch.nn.Module) -> list[dict[str, float]]:
    rows: list[dict[str, float]] = []
    for idx, block in enumerate(getattr(model, "blocks", [])):
        debug = getattr(block, "last_debug", {}) or {}
        rows.append({
            "block": float(idx),
            "style_gate_value": _to_float(debug.get("style_gate_value", 0.0)),
            "cross_attn_delta_abs": _to_float(debug.get("cross_attn_delta_abs", 0.0)),
            "ca_input_std": _to_float(debug.get("ca_input_std", 0.0)),
            "ca_output_std": _to_float(debug.get("ca_output_std", 0.0)),
            "gate_mean": _to_float(debug.get("gate_mean", 0.0)),
            "gate_std": _to_float(debug.get("gate_std", 0.0)),
        })
    return rows


def _band_sensitivity(base: dict[str, torch.Tensor], changed: dict[str, torch.Tensor]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for band in ("ll", "lh", "hl", "hh"):
        if band not in base or band not in changed:
            continue
        delta = (changed[band] - base[band]).detach().float()
        out[band] = {
            "delta_rms": _rms(delta),
            "delta_abs": _abs_mean(delta),
            "base_rms": _rms(base[band]),
            "delta_over_base": float(_rms(delta) / (_rms(base[band]) + 1e-12)),
        }
    return out


def style_condition_sensitivity(model: torch.nn.Module, batch: dict[str, Any], num_styles: int) -> dict[str, Any]:
    model.eval()
    n_id = min(int(num_styles), 5)
    content_id = batch["content"][:1].detach().expand(n_id, -1, -1, -1).contiguous()
    fixed_style_latent_id = batch["target_style"][:1].detach().expand(n_id, -1, -1, -1).contiguous()
    style_ids = torch.arange(n_id, device=content_id.device, dtype=torch.long)
    fixed_style_id = torch.zeros((n_id,), device=content_id.device, dtype=torch.long)
    t_id = torch.full((n_id,), 0.5, device=content_id.device, dtype=content_id.dtype)

    n_latent = min(int(batch["target_style"].shape[0]), 5)
    content_latent = batch["content"][:1].detach().expand(n_latent, -1, -1, -1).contiguous()
    fixed_style_latent = batch["target_style"][:1].detach().expand(n_latent, -1, -1, -1).contiguous()
    style_latents = batch["target_style"][:n_latent].detach().contiguous()
    fixed_style_id_latent = torch.zeros((n_latent,), device=content_latent.device, dtype=torch.long)
    t_latent = torch.full((n_latent,), 0.5, device=content_latent.device, dtype=content_latent.dtype)
    with torch.no_grad():
        base_id = model(content_id, t=t_id, style_id=fixed_style_id, style_latent=fixed_style_latent_id)
        id_changed = model(content_id, t=t_id, style_id=style_ids, style_latent=fixed_style_latent_id)
        base_latent = model(content_latent, t=t_latent, style_id=fixed_style_id_latent, style_latent=fixed_style_latent)
        latent_changed = model(content_latent, t=t_latent, style_id=fixed_style_id_latent, style_latent=style_latents)
        both_changed = model(content_latent, t=t_latent, style_id=style_ids[:n_latent], style_latent=style_latents)
    out = {
        "style_id_only_fixed_target_latent": _band_sensitivity(base_id, id_changed),
        "target_style_latent_only_fixed_id": _band_sensitivity(base_latent, latent_changed),
        "style_id_and_target_latent": _band_sensitivity(base_latent, both_changed),
    }
    model.train()
    return out


def summarize_markdown(results: dict[str, Any]) -> str:
    lines = [
        "# Baseline Internal Flow Probe",
        "",
        f"Config: `{results['config']}`",
        f"Checkpoint: `{results['checkpoint']}`",
        f"Device: `{results['device']}`",
        f"Batches: {results['num_batches']}, batch size: {results['batch_size']}",
        f"Load info: `{results['load_info']}`",
        "",
        "## Loss Components",
        "",
        "| component | value | weighted value |",
        "|---|---:|---:|",
    ]
    for key, value in results["loss_components"].items():
        weighted = results["weighted_loss_components"].get(key, value)
        lines.append(f"| {key} | {value:.6f} | {weighted:.6f} |")

    lines.extend(["", "## Parameter Gradient Groups", ""])
    lines.extend(["| group | grad norm | grad/param |", "|---|---:|---:|"])
    grad_rows = sorted(
        results["grad_summary"].items(),
        key=lambda item: item[1]["grad_norm"],
        reverse=True,
    )
    for name, row in grad_rows:
        lines.append(f"| {name} | {row['grad_norm']:.6e} | {row['grad_over_param']:.6e} |")

    lines.extend(["", "## Aggregated Gradient Paths", ""])
    lines.extend(["| path | grad norm | grad/param |", "|---|---:|---:|"])
    for name, row in results["path_grad_summary"].items():
        lines.append(f"| {name} | {row['grad_norm']:.6e} | {row['grad_over_param']:.6e} |")

    lines.extend(["", "## Input Tensor Gradients", ""])
    lines.extend(["| tensor | tensor rms | grad rms | grad/tensor |", "|---|---:|---:|---:|"])
    for name, row in results["input_grad_summary"].items():
        lines.append(
            f"| {name} | {row['tensor_rms']:.6e} | {row['grad_rms']:.6e} | "
            f"{row['grad_over_tensor']:.6e} |"
        )

    lines.extend(["", "## Per-Loss Gradient Paths", ""])
    for loss_name, paths in results["per_loss_path_gradients"].items():
        lines.extend([f"", f"### {loss_name}", ""])
        lines.extend(["| path | grad norm | grad/param |", "|---|---:|---:|"])
        for path_name, row in paths.items():
            lines.append(f"| {path_name} | {row['grad_norm']:.6e} | {row['grad_over_param']:.6e} |")

    if results.get("loss_gradient_cosines"):
        lines.extend(["", "## Loss Gradient Cosines", ""])
        lines.extend(["| pair / norm | value |", "|---|---:|"])
        for name, value in results["loss_gradient_cosines"].items():
            lines.append(f"| {name} | {value:.6e} |")

    lines.extend(["", "## Activation Gradient Probes", ""])
    lines.extend(["| module | act rms | grad rms | grad/act |", "|---|---:|---:|---:|"])
    for name, row in results["activation_summary"].items():
        act = row.get("act_rms", 0.0)
        grad = row.get("grad_rms", 0.0)
        lines.append(f"| {name} | {act:.6e} | {grad:.6e} | {grad / (act + 1e-12):.6e} |")

    lines.extend(["", "## Cross-Attention Debug", ""])
    lines.extend(["| block | style gate | delta abs | ca in std | ca out std |", "|---:|---:|---:|---:|---:|"])
    for row in results["block_debug"]:
        lines.append(
            f"| {int(row['block'])} | {row['style_gate_value']:.6f} | "
            f"{row['cross_attn_delta_abs']:.6f} | {row['ca_input_std']:.6f} | "
            f"{row['ca_output_std']:.6f} |"
        )

    lines.extend(["", "## Style Condition Sensitivity", ""])
    for condition_name, bands in results["style_condition_sensitivity"].items():
        lines.extend([f"", f"### {condition_name}", ""])
        lines.extend(["| band | delta rms | base rms | delta/base |", "|---|---:|---:|---:|"])
        for band, row in bands.items():
            lines.append(
                f"| {band} | {row['delta_rms']:.6e} | "
                f"{row['base_rms']:.6e} | {row['delta_over_base']:.6e} |"
            )
    lines.append("")
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "exp_brk_a_ll03_10ep.json")
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "exp" / "dino_s_break" / "brk_a_ll03_10ep" / "epoch_0010.pt")
    parser.add_argument("--output", type=Path, default=ROOT / "docs" / "model_probe" / "baseline_internal_flow.json")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-batches", type=int, default=4)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument("--latent-cache-mode", default="off", choices=["off", "manifest", "packed", "refresh"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--enable-hf-stat-loss", action="store_true")
    parser.add_argument("--hf-stat-weight", type=float, default=2.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(int(args.seed))
    device = torch.device(args.device)
    config = load_experiment_config(args.config)
    dataset = build_dataset(config, args.batch_size, args.data_root, args.latent_cache_mode)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    model = build_and_load_model(config, args.checkpoint, device)
    model.train()
    loss_fn = FlowMatchingObjective(config)
    if args.enable_hf_stat_loss:
        loss_fn.hf_stat_loss_enabled = True
        loss_fn.hf_stat_weight = float(args.hf_stat_weight)
    probe = attach_activation_probe(model)

    loss_accum: dict[str, list[float]] = defaultdict(list)
    weighted_accum: dict[str, list[float]] = defaultdict(list)
    last_batch: dict[str, Any] | None = None
    for batch_idx, raw_batch in enumerate(dataloader, start=1):
        if batch_idx > args.num_batches:
            break
        batch = move_batch(raw_batch, device)
        batch["content"] = batch["content"].detach().requires_grad_(True)
        batch["target_style"] = batch["target_style"].detach().requires_grad_(True)
        last_batch = batch
        model.zero_grad(set_to_none=True)
        metrics = loss_fn.compute(
            model,
            content=batch["content"],
            target_style=batch["target_style"],
            target_style_id=batch["target_style_id"],
            source_style_id=batch.get("source_style_id"),
            conditioning=batch,
        )
        loss = metrics["loss"]
        loss.backward()
        for key, value in metrics.items():
            if key == "loss" or key.startswith("loss_fm_spectral") or key in {"flow", "stat", "fft", "t_mean"}:
                loss_accum[key].append(_to_float(value))
        weighted_accum["loss_fm_spectral_ll"].append(float(getattr(config.bridge, "spectral_w_ll", 0.0)) * _to_float(metrics["loss_fm_spectral_ll"]))
        weighted_accum["loss_fm_spectral_lh"].append(float(getattr(config.bridge, "spectral_w_lh", 1.0)) * _to_float(metrics["loss_fm_spectral_lh"]))
        weighted_accum["loss_fm_spectral_hl"].append(float(getattr(config.bridge, "spectral_w_hl", 1.0)) * _to_float(metrics["loss_fm_spectral_hl"]))
        weighted_accum["loss_fm_spectral_hh"].append(float(getattr(config.bridge, "spectral_w_hh", 2.0)) * _to_float(metrics["loss_fm_spectral_hh"]))

    if last_batch is None:
        raise RuntimeError("No batches were available for probe.")

    grad_summary = collect_grad_summary(model)
    path_grad_summary = aggregate_path_summary(grad_summary)
    input_grad_summary = collect_input_grad_summary(last_batch)
    activation_summary = probe.summary()
    block_debug = collect_block_debug(model)
    probe.close()
    per_loss_path_gradients = collect_loss_path_gradients(model, loss_fn, last_batch, config)
    loss_gradient_cosines = collect_loss_gradient_cosines(model, loss_fn, last_batch, config)
    style_sens = style_condition_sensitivity(model, last_batch, int(getattr(config.model, "num_styles", 5)))

    results = {
        "config": str(args.config),
        "checkpoint": str(args.checkpoint),
        "device": str(device),
        "num_batches": int(args.num_batches),
        "batch_size": int(args.batch_size),
        "data_root": getattr(dataset, "_probe_data_root", ""),
        "latent_cache_mode": getattr(dataset, "_probe_latent_cache_mode", ""),
        "load_info": getattr(model, "_probe_load_info", {}),
        "model_focus": {
            "structure_aligned_target": bool(getattr(config.bridge, "structure_aligned_target", False)),
            "ll_partial_style_enabled": bool(getattr(config.bridge, "ll_partial_style_enabled", False)),
            "ll_partial_alpha": float(getattr(config.bridge, "ll_partial_alpha", 0.0)),
            "endpoint_adain_scale": float(getattr(config.model, "endpoint_adain_scale", 0.0)),
            "cross_attn_dwt_route": bool(getattr(config.model, "cross_attn_dwt_route", False)),
            "enable_hh_head": bool(getattr(config.model, "enable_hh_head", False)),
            "probe_hf_stat_loss_enabled": bool(loss_fn.hf_stat_loss_enabled),
            "probe_hf_stat_weight": float(loss_fn.hf_stat_weight),
        },
        "loss_components": {key: _mean(values) for key, values in loss_accum.items()},
        "weighted_loss_components": {key: _mean(values) for key, values in weighted_accum.items()},
        "grad_summary": grad_summary,
        "path_grad_summary": path_grad_summary,
        "input_grad_summary": input_grad_summary,
        "per_loss_path_gradients": per_loss_path_gradients,
        "loss_gradient_cosines": loss_gradient_cosines,
        "activation_summary": activation_summary,
        "block_debug": block_debug,
        "style_condition_sensitivity": style_sens,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2), encoding="utf-8")
    args.output.with_suffix(".md").write_text(summarize_markdown(results), encoding="utf-8")
    print(f"Wrote {args.output}")
    print(f"Wrote {args.output.with_suffix('.md')}")


if __name__ == "__main__":
    main()
