from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Iterable

import torch


def _module_params(module: torch.nn.Module | None) -> list[torch.nn.Parameter]:
    return list(module.parameters()) if module is not None else []


def _unique_trainable(params: Iterable[torch.nn.Parameter]) -> list[torch.nn.Parameter]:
    unique: list[torch.nn.Parameter] = []
    seen: set[int] = set()
    for param in params:
        if not isinstance(param, torch.nn.Parameter) or not param.requires_grad or id(param) in seen:
            continue
        seen.add(id(param))
        unique.append(param)
    return unique


def _grad_norm(params: Iterable[torch.nn.Parameter]) -> float:
    total = 0.0
    for param in params:
        if param.grad is None:
            continue
        grad = torch.nan_to_num(param.grad.detach().float())
        total += float(grad.square().sum().item())
    return math.sqrt(total)


def _shared_trunk_params(model: torch.nn.Module) -> list[torch.nn.Parameter]:
    params: list[torch.nn.Parameter] = []
    params += _module_params(getattr(model, "input_proj", None))
    params += _module_params(getattr(model, "time_proj", None))
    for block in getattr(model, "blocks", []):
        params += _module_params(getattr(block, "sa_qkv", None))
        params += _module_params(getattr(block, "sa_out", None))
        params += _module_params(getattr(block, "q_proj", None))
        params += _module_params(getattr(block, "k_proj", None))
        params += _module_params(getattr(block, "v_proj", None))
        params += _module_params(getattr(block, "out_proj", None))
        params += _module_params(getattr(block, "time_style_adaln", getattr(block, "time_adaln", None)))
        params += _module_params(getattr(block, "ffn", None))
        style_gate = getattr(block, "style_gate", None)
        if isinstance(style_gate, torch.nn.Parameter):
            params.append(style_gate)
    return _unique_trainable(params)


def _target_hf_route_params(model: torch.nn.Module) -> list[torch.nn.Parameter]:
    params: list[torch.nn.Parameter] = []
    names = (
        "target_latent_hf_subband_encoder_lh",
        "target_latent_hf_subband_encoder_hl",
        "target_latent_hf_subband_encoder_hh",
        "target_latent_hf_subband_proj_lh",
        "target_latent_hf_subband_proj_hl",
        "target_latent_hf_subband_proj_hh",
        "target_latent_hf_subband_delta_lh",
        "target_latent_hf_subband_delta_hl",
        "target_latent_hf_subband_delta_hh",
    )
    for name in names:
        params += _module_params(getattr(model, name, None))
    gate = getattr(model, "target_latent_hf_subband_head_gate", None)
    if isinstance(gate, torch.nn.Parameter):
        params.append(gate)
    return _unique_trainable(params)


def _hf_head_params(model: torch.nn.Module) -> list[torch.nn.Parameter]:
    params: list[torch.nn.Parameter] = []
    for name in ("head_lh", "head_hl", "head_hh"):
        params += _module_params(getattr(model, name, None))
    return _unique_trainable(params)


def target_hf_gate_mean(model: torch.nn.Module) -> float:
    gates: list[torch.Tensor] = []
    for name in (
        "target_latent_hf_subband_delta_lh",
        "target_latent_hf_subband_delta_hl",
    ):
        module = getattr(model, name, None)
        gate = getattr(module, "gate", None)
        if isinstance(gate, torch.nn.Parameter):
            gates.append(torch.tanh(gate.detach().float()))
    if not gates:
        return 0.0
    return float(torch.stack(gates).mean().item())


def probe_internal_dynamics(
    model: torch.nn.Module,
    loss_fn: Any,
    batch: dict[str, Any],
    *,
    fixed_t: float = 0.5,
    noise: torch.Tensor | None = None,
) -> dict[str, float]:
    """Measure two loss-path gradients on a fixed latent batch."""
    content = batch["content"]
    target_style = batch["target_style"]
    target_style_id = batch["target_style_id"]
    conditioning = {
        key: value
        for key, value in batch.items()
        if key in {"target_style_text_tokens", "target_style_latent"}
    }
    t = torch.full(
        (content.shape[0],),
        float(fixed_t),
        device=content.device,
        dtype=content.dtype,
    )
    shared_params = _shared_trunk_params(model)
    route_params = _target_hf_route_params(model)
    head_params = _hf_head_params(model)

    grad_rows: dict[str, dict[str, float]] = {}
    loss_values: dict[str, float] = {}
    for loss_name in ("loss_fm_spectral_ll", "loss_fm_hf_total"):
        model.zero_grad(set_to_none=True)
        losses = loss_fn.compute_probe_losses(
            model,
            content=content,
            target_style=target_style,
            target_style_id=target_style_id,
            conditioning=conditioning,
            t=t,
            noise=noise,
        )
        loss = losses[loss_name]
        loss.backward()
        loss_values[loss_name] = float(loss.detach().float().item())
        grad_rows[loss_name] = {
            "shared": _grad_norm(shared_params),
            "route": _grad_norm(route_params),
            "head": _grad_norm(head_params),
        }
    model.zero_grad(set_to_none=True)

    ll_shared = grad_rows["loss_fm_spectral_ll"]["shared"]
    hf_shared = grad_rows["loss_fm_hf_total"]["shared"]
    hf_route = grad_rows["loss_fm_hf_total"]["route"]
    hf_head = grad_rows["loss_fm_hf_total"]["head"]
    eps = 1e-12
    return {
        "internal_probe_active": 1.0,
        "internal_probe_gate_mean": target_hf_gate_mean(model),
        "internal_probe_loss_ll": loss_values["loss_fm_spectral_ll"],
        "internal_probe_loss_hf": loss_values["loss_fm_hf_total"],
        "internal_probe_shared_ll_grad_norm": ll_shared,
        "internal_probe_shared_hf_grad_norm": hf_shared,
        "internal_probe_shared_ll_hf_grad_ratio": ll_shared / (hf_shared + eps),
        "internal_probe_target_hf_route_grad_norm": hf_route,
        "internal_probe_hf_head_grad_norm": hf_head,
        "internal_probe_route_shared_hf_grad_ratio": hf_route / (hf_shared + eps),
        "internal_probe_route_hf_head_grad_ratio": hf_route / (hf_head + eps),
    }


@dataclass
class InternalDynamicsState:
    previous_gate_mean: float | None = None
    previous_shared_ll_hf_ratio: float | None = None
    transition_epoch: int | None = None

    def update(
        self,
        epoch: int,
        metrics: dict[str, float],
        *,
        min_epoch: int,
        gate_delta_threshold: float,
        shared_ratio_drop_threshold: float,
    ) -> bool:
        gate_mean = float(metrics["internal_probe_gate_mean"])
        shared_ratio = float(metrics["internal_probe_shared_ll_hf_grad_ratio"])
        gate_delta = 0.0 if self.previous_gate_mean is None else gate_mean - self.previous_gate_mean
        ratio_step = (
            1.0
            if self.previous_shared_ll_hf_ratio is None
            else shared_ratio / max(self.previous_shared_ll_hf_ratio, 1e-12)
        )
        crossed = bool(
            epoch >= int(min_epoch)
            and self.previous_shared_ll_hf_ratio is not None
            and ratio_step <= float(shared_ratio_drop_threshold)
            and gate_delta > float(gate_delta_threshold)
        )
        if crossed and self.transition_epoch is None:
            self.transition_epoch = int(epoch)
        metrics["internal_probe_gate_delta"] = gate_delta
        metrics["internal_probe_shared_ratio_step"] = ratio_step
        metrics["internal_probe_transition"] = float(crossed)
        metrics["internal_probe_transition_epoch"] = float(self.transition_epoch or 0)
        self.previous_gate_mean = gate_mean
        self.previous_shared_ll_hf_ratio = shared_ratio
        return crossed
