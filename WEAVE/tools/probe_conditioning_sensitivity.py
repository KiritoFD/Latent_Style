from __future__ import annotations

import argparse
import copy
import csv
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import torch


def _repo_src_path() -> Path:
    return Path(__file__).resolve().parents[1] / "src"


SRC_PATH = str(_repo_src_path())
if SRC_PATH not in sys.path:
    sys.path.insert(0, SRC_PATH)

from config_schema import ExperimentConfig, load_experiment_config  # noqa: E402
from lancet_blocks import StyleMaps  # noqa: E402
from model import build_model_from_config  # noqa: E402
from utils.training import strip_compile_prefix  # noqa: E402


def _git_commit() -> str | None:
    repo_root = Path(__file__).resolve().parents[1]
    try:
        result = subprocess.run(
            ["git", "-C", str(repo_root), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except Exception:
        return None
    return result.stdout.strip() or None


def _runtime_metadata(device_request: str, device: torch.device) -> dict[str, Any]:
    meta: dict[str, Any] = {
        "device_request": device_request,
        "device": str(device),
        "torch_version": torch.__version__,
    }
    if device.type == "cuda":
        meta.update(
            {
                "cuda_device_name": torch.cuda.get_device_name(device),
                "cuda_capability": list(torch.cuda.get_device_capability(device)),
            }
        )
    return meta


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _attention_entropy(attn: torch.Tensor | None) -> float:
    if attn is None or not torch.is_tensor(attn):
        return 0.0
    probs = attn.detach().float().clamp_min(1e-8)
    entropy = -(probs * probs.log()).sum(dim=-1).mean()
    return float(entropy.item())


def _mean_abs_diff(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a.detach() - b.detach()).float().abs().mean().item())


def _style_path_debug_fields(style_path_debug: dict[str, Any]) -> dict[str, Any]:
    return {
        "style_spatial_source_override_palette": float(style_path_debug.get("style_spatial_source_override_palette", 0.0)),
        "style_spatial_source_target_latent": float(style_path_debug.get("style_spatial_source_target_latent", 0.0)),
        "style_spatial_source_structured_map": float(style_path_debug.get("style_spatial_source_structured_map", 0.0)),
        "style_spatial_source_code_map": float(style_path_debug.get("style_spatial_source_code_map", 0.0)),
        "style_spatial_source_legacy_zero": float(style_path_debug.get("style_spatial_source_legacy_zero", 0.0)),
        "style_spatial_code_map_primary": float(style_path_debug.get("style_spatial_code_map_primary", 0.0)),
        "style_spatial_code_map_residual": float(style_path_debug.get("style_spatial_code_map_residual", 0.0)),
        "style_spatial_code_map_pre_resolved_abs": float(
            style_path_debug.get("style_spatial_code_map_pre_resolved_abs", 0.0)
        ),
        "style_spatial_code_map_abs": float(style_path_debug.get("style_spatial_code_map_abs", 0.0)),
        "style_spatial_map_abs": float(style_path_debug.get("style_spatial_map_abs", 0.0)),
    }


def _clone_config(cfg: ExperimentConfig) -> ExperimentConfig:
    return copy.deepcopy(cfg)


def _parse_override_value(raw: str) -> Any:
    text = str(raw).strip()
    if text == "":
        return text
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        lowered = text.lower()
        if lowered == "true":
            return True
        if lowered == "false":
            return False
        if lowered == "null":
            return None
        return text


def _apply_config_overrides(cfg: ExperimentConfig, overrides: list[str]) -> dict[str, Any]:
    applied: dict[str, Any] = {}
    for entry in overrides:
        if "=" not in entry:
            raise ValueError(f"Override must be KEY=VALUE, got: {entry!r}")
        key, raw_value = entry.split("=", 1)
        dotted = key.strip()
        if not dotted or "." not in dotted:
            raise ValueError(f"Override key must be section.field, got: {dotted!r}")
        section_name, field_name = dotted.split(".", 1)
        section = getattr(cfg, section_name, None)
        if section is None:
            raise AttributeError(f"ExperimentConfig has no section {section_name!r}")
        value = _parse_override_value(raw_value)
        setattr(section, field_name, value)
        applied[dotted] = value
    return applied


def _load_checkpoint_state(checkpoint: Path) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    state: dict[str, torch.Tensor] | None = None
    if isinstance(payload, dict):
        for key in ("model_state_dict", "state_dict", "model"):
            candidate = payload.get(key)
            if isinstance(candidate, dict):
                state = candidate
                break
        if state is None and all(isinstance(k, str) for k in payload.keys()):
            state = payload
    if state is None:
        raise TypeError(f"Unsupported checkpoint payload: {checkpoint}")
    clean = strip_compile_prefix({str(k): v for k, v in state.items()})
    return clean, {"checkpoint": str(checkpoint), "num_tensors": len(clean)}


def _build_model(
    cfg: ExperimentConfig,
    *,
    device: torch.device,
    seed: int,
    checkpoint: Path | None,
) -> tuple[torch.nn.Module, dict[str, Any]]:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    model = build_model_from_config(cfg.model, bridge_cfg=cfg.bridge, use_checkpointing=False)
    checkpoint_meta: dict[str, Any] = {}
    if checkpoint is not None:
        state_dict, checkpoint_meta = _load_checkpoint_state(checkpoint)
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        checkpoint_meta = {
            **checkpoint_meta,
            "missing_keys": list(missing),
            "unexpected_keys": list(unexpected),
        }
    model = model.to(device=device)
    model.eval()
    return model, checkpoint_meta


def _random_inputs(
    *,
    batch_size: int,
    latent_channels: int,
    latent_size: int,
    style_id: int,
    seed: int,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    x = torch.randn(batch_size, latent_channels, latent_size, latent_size, generator=generator)
    lat_a = torch.randn(batch_size, latent_channels, latent_size, latent_size, generator=generator)
    lat_b = torch.randn(batch_size, latent_channels, latent_size, latent_size, generator=generator)
    t = torch.full((batch_size,), 0.5, dtype=torch.float32)
    style = torch.full((batch_size,), int(style_id), dtype=torch.long)
    return {
        "x": x.to(device=device),
        "lat_a": lat_a.to(device=device),
        "lat_b": lat_b.to(device=device),
        "t": t.to(device=device),
        "style_id": style.to(device=device),
    }


def _conditioning_payload(
    model: torch.nn.Module,
    *,
    mode: str,
    style_id: torch.Tensor,
    latent: torch.Tensor,
) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    target_style_latent = latent if mode in {"spatial", "both"} else None
    style_code_override = None
    if mode in {"code", "both"}:
        style_code_override = model.encode_target_style_latent(latent, style_id=style_id)
    return target_style_latent, style_code_override


def _probe_triplet(
    model: torch.nn.Module,
    *,
    x: torch.Tensor,
    t: torch.Tensor,
    style_id: torch.Tensor,
    target_style_latent: torch.Tensor | None,
    style_code_override: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    with torch.no_grad():
        out = model(
            x,
            t=t,
            style_id=style_id,
            target_style_latent=target_style_latent,
            style_code_override=style_code_override,
        )
        base = model.predict_transport_base(
            x,
            t=t,
            style_id=style_id,
            target_style_latent=target_style_latent,
            style_code_override=style_code_override,
        )
        end = model.integrate(
            x,
            style_id=style_id,
            num_steps=2,
            step_size=1.0,
            target_style_latent=target_style_latent,
            style_code_override=style_code_override,
        )
    return out, base, end


def _conditioning_rows(
    base_cfg: ExperimentConfig,
    *,
    device: torch.device,
    seed: int,
    checkpoint: Path | None,
    inputs: dict[str, torch.Tensor],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    checkpoint_meta: dict[str, Any] = {}
    modes = [
        ("none", "none"),
        ("spatial", "none"),
        ("code", "residual"),
        ("both", "residual"),
    ]
    for mode, encoder_mode in modes:
        cfg = _clone_config(base_cfg)
        cfg.model.matched_target_conditioning_mode = mode
        cfg.model.matched_target_style_encoder_mode = encoder_mode
        cfg.model.matched_target_style_encoder_hidden_dim = max(
            64,
            int(getattr(cfg.model, "matched_target_style_encoder_hidden_dim", 64)),
        )
        model, checkpoint_meta = _build_model(cfg, device=device, seed=seed, checkpoint=checkpoint)
        target_a, code_a = _conditioning_payload(
            model,
            mode=mode,
            style_id=inputs["style_id"],
            latent=inputs["lat_a"],
        )
        target_b, code_b = _conditioning_payload(
            model,
            mode=mode,
            style_id=inputs["style_id"],
            latent=inputs["lat_b"],
        )
        out_a, base_a, end_a = _probe_triplet(
            model,
            x=inputs["x"],
            t=inputs["t"],
            style_id=inputs["style_id"],
            target_style_latent=target_a,
            style_code_override=code_a,
        )
        out_b, base_b, end_b = _probe_triplet(
            model,
            x=inputs["x"],
            t=inputs["t"],
            style_id=inputs["style_id"],
            target_style_latent=target_b,
            style_code_override=code_b,
        )
        style_path_debug = getattr(model, "last_style_path_debug", {}) or {}
        style_code_debug = getattr(model, "last_style_code_path_debug", {}) or {}
        cached_ctx = getattr(model, "last_output_style_context", None)
        cached_map = None
        if isinstance(cached_ctx, dict):
            cached_maps = cached_ctx.get("style_maps")
            cached_map = getattr(cached_maps, "map_16", None) if cached_maps is not None else None
        code_delta = _mean_abs_diff(code_a, code_b) if torch.is_tensor(code_a) and torch.is_tensor(code_b) else 0.0
        rows.append(
            {
                "probe_family": "conditioning",
                "mode": mode,
                "encoder_mode": encoder_mode,
                "spatial_active": float(target_a is not None),
                "code_active": float(code_a is not None),
                "encoded_code_a_vs_b_mean_abs": code_delta,
                "forward_a_vs_b_mean_abs": _mean_abs_diff(out_a, out_b),
                "predict_transport_base_a_vs_b_mean_abs": _mean_abs_diff(base_a, base_b),
                "integrate_a_vs_b_mean_abs": _mean_abs_diff(end_a, end_b),
                "style_code_override_active": float(style_code_debug.get("style_code_override_active", 0.0)),
                "style_code_content_router_active": float(style_code_debug.get("style_code_content_router_active", 0.0)),
                "style_code_content_router_bypassed": float(style_code_debug.get("style_code_content_router_bypassed", 0.0)),
                "style_code_content_delta_abs": float(style_code_debug.get("style_code_content_delta_abs", 0.0)),
                "style_code_adapted_abs": float(style_code_debug.get("style_code_adapted_abs", 0.0)),
                "cached_output_style_map_present": float(torch.is_tensor(cached_map)),
                "cached_output_style_map_abs": (
                    float(cached_map.detach().float().abs().mean().item()) if torch.is_tensor(cached_map) else 0.0
                ),
                **_style_path_debug_fields(style_path_debug),
            }
        )
    summary = {
        "conditioning_code_forward_live": rows[2]["forward_a_vs_b_mean_abs"] > 0.0,
        "conditioning_spatial_forward_delta": rows[1]["forward_a_vs_b_mean_abs"],
        "conditioning_code_forward_delta": rows[2]["forward_a_vs_b_mean_abs"],
        "conditioning_both_forward_delta": rows[3]["forward_a_vs_b_mean_abs"],
        "checkpoint_meta": checkpoint_meta,
    }
    return rows, summary


def _topology_rows(
    base_cfg: ExperimentConfig,
    *,
    device: torch.device,
    seed: int,
    checkpoint: Path | None,
    inputs: dict[str, torch.Tensor],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    variants = [
        ("gate0_blend0", False, 0.0),
        ("gate0_blend1", False, 1.0),
        ("gate1_blend0", True, 0.0),
        ("gate1_blend05", True, 0.5),
        ("gate1_blend1", True, 1.0),
    ]
    rows: list[dict[str, Any]] = []
    cached_outputs: dict[str, torch.Tensor] = {}
    checkpoint_meta: dict[str, Any] = {}
    for name, gate, blend in variants:
        cfg = _clone_config(base_cfg)
        cfg.model.semantic_self_topology_gate = gate
        cfg.model.semantic_self_topology_blend = blend
        cfg.model.matched_target_conditioning_mode = "spatial"
        cfg.model.matched_target_style_encoder_mode = "none"
        model, checkpoint_meta = _build_model(cfg, device=device, seed=seed, checkpoint=checkpoint)
        out_a, _, _ = _probe_triplet(
            model,
            x=inputs["x"],
            t=inputs["t"],
            style_id=inputs["style_id"],
            target_style_latent=inputs["lat_a"],
            style_code_override=None,
        )
        out_b, _, _ = _probe_triplet(
            model,
            x=inputs["x"],
            t=inputs["t"],
            style_id=inputs["style_id"],
            target_style_latent=inputs["lat_b"],
            style_code_override=None,
        )
        cached_outputs[name] = out_a.detach().cpu()
        attn = getattr(model, "last_semantic_attn", None)
        topo_attn = getattr(model, "last_semantic_topology_attn", None)
        style_path_debug = getattr(model, "last_style_path_debug", {}) or {}
        rows.append(
            {
                "probe_family": "topology",
                "variant": name,
                "semantic_self_topology_gate": float(gate),
                "semantic_self_topology_blend": float(blend),
                "forward_a_vs_b_mean_abs": _mean_abs_diff(out_a, out_b),
                "semantic_attn_entropy": _attention_entropy(attn),
                "semantic_topology_attn_active": float(topo_attn is not None),
                "semantic_topology_attn_entropy": _attention_entropy(topo_attn),
                **_style_path_debug_fields(style_path_debug),
            }
        )
    pair_rows = []
    pairs = [
        ("gate0_blend0", "gate0_blend1"),
        ("gate1_blend0", "gate1_blend05"),
        ("gate1_blend0", "gate1_blend1"),
        ("gate0_blend1", "gate1_blend1"),
    ]
    for left, right in pairs:
        pair_rows.append(
            {
                "probe_family": "topology_pair",
                "left_variant": left,
                "right_variant": right,
                "forward_same_target_mean_abs": _mean_abs_diff(cached_outputs[left], cached_outputs[right]),
            }
        )
    summary = {
        "topology_gate0_blend_noop_delta": pair_rows[0]["forward_same_target_mean_abs"],
        "topology_gate1_blend_effect_delta": pair_rows[2]["forward_same_target_mean_abs"],
        "topology_gate1_half_vs_zero_delta": pair_rows[1]["forward_same_target_mean_abs"],
        "topology_gate1_style_sensitivity_blend0": rows[2]["forward_a_vs_b_mean_abs"],
        "topology_gate1_style_sensitivity_blend1": rows[4]["forward_a_vs_b_mean_abs"],
        "checkpoint_meta": checkpoint_meta,
    }
    return rows, pair_rows, summary


def _first_live_stage(row: dict[str, Any]) -> tuple[str, float]:
    ordered = [
        "first_hires_block_gate1_a_vs_b_mean_abs",
        "skip32_a_vs_b_mean_abs",
        "content16_a_vs_b_mean_abs",
        "adapted_code_a_vs_b_mean_abs",
        "style_map_a_vs_b_mean_abs",
        "h_body_a_vs_b_mean_abs",
        "h_fused_a_vs_b_mean_abs",
        "h_dec_pre_mod_a_vs_b_mean_abs",
        "h_dec_post_mod_a_vs_b_mean_abs",
        "delta_a_vs_b_mean_abs",
    ]
    for key in ordered:
        value = float(row.get(key, 0.0) or 0.0)
        if value > 0.0:
            return key, value
    return "none", 0.0


def _trace_runtime_delta_path(
    model: torch.nn.Module,
    *,
    x: torch.Tensor,
    t: torch.Tensor,
    style_id: torch.Tensor,
    target_style_latent: torch.Tensor | None = None,
    style_code_override: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    feat_c = x / max(model.latent_scale_factor, 1e-8)
    h_c = model.enc_in_act(model.enc_in(feat_c))
    style_code = model._compute_style_code(
        x=x,
        style_id=style_id,
        t=t,
        style_code_override=style_code_override,
    )
    first_block_out = h_c
    if len(model.hires_body) > 0:
        first_block_out = model._run_block(model.hires_body[0], h_c.clone(), style_code, gate=1.0, shift=False)

    h_c_grad = h_c
    for block in model.hires_body:
        h_c_grad = block(h_c_grad, style_code, gate=0.0)
    skip_32 = h_c_grad.detach()
    content_feat_16 = model.down(h_c_grad)
    adapted_code = model._adapt_style_code_from_content(
        style_id=style_id,
        style_code=style_code,
        content_feat_16=content_feat_16,
        style_code_override_active=style_code_override is not None,
    )
    pre_resolved_style_code_map = model._decode_style_code_spatial_map(
        adapted_code,
        target_hw=tuple(int(v) for v in content_feat_16.shape[-2:]),
        device=content_feat_16.device,
        dtype=content_feat_16.dtype,
    )
    style_maps = model._prepare_style_maps(style_id)
    resolved_style_code = adapted_code
    structured_ctx = model._structured_style_from_sidecar(
        style_id=style_id,
        style_code=adapted_code,
        content_latent=x,
        content_feat_16=content_feat_16,
    )
    if structured_ctx is not None:
        resolved_style_code, style_maps = structured_ctx
    style_code_map = model._decode_style_code_spatial_map(
        resolved_style_code,
        target_hw=tuple(int(v) for v in content_feat_16.shape[-2:]),
        device=content_feat_16.device,
        dtype=content_feat_16.dtype,
    )

    style_map = model._prepare_spatial_map(style_maps.map_16, content_feat_16)
    if target_style_latent is not None:
        feat_s = target_style_latent / max(model.latent_scale_factor, 1e-8)
        h_s = model.enc_in_act(model.enc_in(feat_s))
        h_s = model._run_style_blocks(
            h_s,
            blocks=model.hires_body,
            style_code=resolved_style_code,
            base_idx=0,
            gate_scale=0.0,
        )
        style_map_proj = model.down(h_s)
        if style_code_map is not None:
            style_map_proj = style_map_proj + style_code_map
    elif style_map is None:
        if style_code_map is not None:
            style_map_proj = style_code_map
        else:
            style_map_proj = torch.zeros_like(content_feat_16)
    else:
        style_map_proj = style_map
        if style_maps.mask_16 is not None:
            mask_16 = model._prepare_spatial_map(style_maps.mask_16, content_feat_16)
            if mask_16 is not None:
                style_map_proj = style_map_proj * (0.5 + torch.sigmoid(mask_16))
        if style_code_map is not None:
            style_map_proj = style_map_proj + style_code_map

    body_gate: float | torch.Tensor = 1.0
    if style_maps.gate_16 is not None:
        gate_16 = model._prepare_spatial_map(style_maps.gate_16, content_feat_16)
        if gate_16 is not None:
            body_gate = torch.sigmoid(gate_16)

    h_body = content_feat_16
    for block in model.body_blocks:
        h_body = block(h_body, style_map=style_map_proj, gate=body_gate)
    style_inject = getattr(model, "_apply_style_feature_injection", None)
    if callable(style_inject):
        h_body = style_inject(h_body, x, resolved_style_code, site="body", style_map=style_map_proj)

    h_up = model.dec_up(h_body)
    h_up = model._apply_upsample_blur(h_up)
    h_fused = model._fuse_skip_features(h_up, skip_32, style_code=resolved_style_code, gate=1.0)
    h_dec = model._run_decoder(h_fused)
    if callable(style_inject):
        h_dec = style_inject(h_dec, x, resolved_style_code, site="decoder", style_map=style_map_proj)
    h_mod = model.dec_act(model.dec_mod(h_dec, resolved_style_code, gate=1.0))
    delta = model._compute_delta(h_mod, x=x, style_code=resolved_style_code, style_maps=style_maps)
    return {
        "effective_style_code": style_code,
        "first_block_out": first_block_out,
        "skip_32": skip_32,
        "content_feat_16": content_feat_16,
        "adapted_code": adapted_code,
        "resolved_code": resolved_style_code,
        "pre_resolved_style_code_map": pre_resolved_style_code_map,
        "style_map": style_map_proj,
        "h_body": h_body,
        "h_fused": h_fused,
        "h_dec": h_dec,
        "h_mod": h_mod,
        "delta": delta,
    }


def _trace_spatial_matched_target_path(
    model: torch.nn.Module,
    *,
    x: torch.Tensor,
    t: torch.Tensor,
    style_id: torch.Tensor,
    target_style_latent: torch.Tensor,
) -> dict[str, torch.Tensor]:
    return _trace_runtime_delta_path(
        model,
        x=x,
        t=t,
        style_id=style_id,
        target_style_latent=target_style_latent,
        style_code_override=None,
    )


def _trace_code_only_no_reference_path(
    model: torch.nn.Module,
    *,
    x: torch.Tensor,
    t: torch.Tensor,
    style_id: torch.Tensor,
    code: torch.Tensor,
) -> dict[str, torch.Tensor]:
    return _trace_runtime_delta_path(
        model,
        x=x,
        t=t,
        style_id=style_id,
        target_style_latent=None,
        style_code_override=code,
    )


def _path_anatomy_rows(
    base_cfg: ExperimentConfig,
    *,
    device: torch.device,
    seed: int,
    checkpoint: Path | None,
    inputs: dict[str, torch.Tensor],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    code_cfg = _clone_config(base_cfg)
    code_cfg.model.matched_target_conditioning_mode = "code"
    code_cfg.model.matched_target_style_encoder_mode = "residual"
    code_cfg.model.matched_target_style_encoder_hidden_dim = max(
        64,
        int(getattr(code_cfg.model, "matched_target_style_encoder_hidden_dim", 64)),
    )
    code_model, _ = _build_model(code_cfg, device=device, seed=seed, checkpoint=checkpoint)
    with torch.no_grad():
        code_a = code_model.encode_target_style_latent(inputs["lat_a"], style_id=inputs["style_id"])
        code_b = code_model.encode_target_style_latent(inputs["lat_b"], style_id=inputs["style_id"])
        trace_code_a = _trace_code_only_no_reference_path(
            code_model,
            x=inputs["x"],
            t=inputs["t"],
            style_id=inputs["style_id"],
            code=code_a,
        )
        trace_code_b = _trace_code_only_no_reference_path(
            code_model,
            x=inputs["x"],
            t=inputs["t"],
            style_id=inputs["style_id"],
            code=code_b,
        )
    code_row = {
        "probe_family": "path_anatomy",
        "path_mode": "code_only_no_reference",
        "encoded_code_a_vs_b_mean_abs": _mean_abs_diff(code_a, code_b),
        "first_hires_block_gate1_a_vs_b_mean_abs": _mean_abs_diff(trace_code_a["first_block_out"], trace_code_b["first_block_out"]),
        "skip32_a_vs_b_mean_abs": _mean_abs_diff(trace_code_a["skip_32"], trace_code_b["skip_32"]),
        "content16_a_vs_b_mean_abs": _mean_abs_diff(trace_code_a["content_feat_16"], trace_code_b["content_feat_16"]),
        "adapted_code_a_vs_b_mean_abs": _mean_abs_diff(trace_code_a["adapted_code"], trace_code_b["adapted_code"]),
        "style_map_a_vs_b_mean_abs": _mean_abs_diff(trace_code_a["style_map"], trace_code_b["style_map"]),
        "h_body_a_vs_b_mean_abs": _mean_abs_diff(trace_code_a["h_body"], trace_code_b["h_body"]),
        "h_fused_a_vs_b_mean_abs": _mean_abs_diff(trace_code_a["h_fused"], trace_code_b["h_fused"]),
        "h_dec_pre_mod_a_vs_b_mean_abs": _mean_abs_diff(trace_code_a["h_dec"], trace_code_b["h_dec"]),
        "h_dec_post_mod_a_vs_b_mean_abs": _mean_abs_diff(trace_code_a["h_mod"], trace_code_b["h_mod"]),
        "delta_a_vs_b_mean_abs": _mean_abs_diff(trace_code_a["delta"], trace_code_b["delta"]),
    }
    rows.append(code_row)

    spatial_cfg = _clone_config(base_cfg)
    spatial_cfg.model.matched_target_conditioning_mode = "spatial"
    spatial_cfg.model.matched_target_style_encoder_mode = "none"
    spatial_model, _ = _build_model(spatial_cfg, device=device, seed=seed, checkpoint=checkpoint)
    with torch.no_grad():
        trace_spatial_a = _trace_spatial_matched_target_path(
            spatial_model,
            x=inputs["x"],
            t=inputs["t"],
            style_id=inputs["style_id"],
            target_style_latent=inputs["lat_a"],
        )
        trace_spatial_b = _trace_spatial_matched_target_path(
            spatial_model,
            x=inputs["x"],
            t=inputs["t"],
            style_id=inputs["style_id"],
            target_style_latent=inputs["lat_b"],
        )
    spatial_row = {
        "probe_family": "path_anatomy",
        "path_mode": "spatial_matched_target",
        "encoded_code_a_vs_b_mean_abs": _mean_abs_diff(
            trace_spatial_a["effective_style_code"],
            trace_spatial_b["effective_style_code"],
        ),
        "first_hires_block_gate1_a_vs_b_mean_abs": _mean_abs_diff(
            trace_spatial_a["first_block_out"],
            trace_spatial_b["first_block_out"],
        ),
        "skip32_a_vs_b_mean_abs": _mean_abs_diff(trace_spatial_a["skip_32"], trace_spatial_b["skip_32"]),
        "content16_a_vs_b_mean_abs": _mean_abs_diff(
            trace_spatial_a["content_feat_16"],
            trace_spatial_b["content_feat_16"],
        ),
        "adapted_code_a_vs_b_mean_abs": _mean_abs_diff(
            trace_spatial_a["adapted_code"],
            trace_spatial_b["adapted_code"],
        ),
        "style_map_a_vs_b_mean_abs": _mean_abs_diff(trace_spatial_a["style_map"], trace_spatial_b["style_map"]),
        "h_body_a_vs_b_mean_abs": _mean_abs_diff(trace_spatial_a["h_body"], trace_spatial_b["h_body"]),
        "h_fused_a_vs_b_mean_abs": _mean_abs_diff(trace_spatial_a["h_fused"], trace_spatial_b["h_fused"]),
        "h_dec_pre_mod_a_vs_b_mean_abs": _mean_abs_diff(trace_spatial_a["h_dec"], trace_spatial_b["h_dec"]),
        "h_dec_post_mod_a_vs_b_mean_abs": _mean_abs_diff(trace_spatial_a["h_mod"], trace_spatial_b["h_mod"]),
        "delta_a_vs_b_mean_abs": _mean_abs_diff(trace_spatial_a["delta"], trace_spatial_b["delta"]),
    }
    rows.append(spatial_row)

    code_first_stage, code_first_delta = _first_live_stage(code_row)
    spatial_first_stage, spatial_first_delta = _first_live_stage(spatial_row)
    summary = {
        "anatomy_code_first_live_stage": code_first_stage,
        "anatomy_code_first_live_stage_delta": code_first_delta,
        "anatomy_spatial_first_live_stage": spatial_first_stage,
        "anatomy_spatial_first_live_stage_delta": spatial_first_delta,
        "anatomy_code_body_dead_spatial_body_live": (
            code_row["h_body_a_vs_b_mean_abs"] <= 0.0 and spatial_row["h_body_a_vs_b_mean_abs"] > 0.0
        ),
        "anatomy_code_only_delta": code_row["delta_a_vs_b_mean_abs"],
        "anatomy_spatial_delta": spatial_row["delta_a_vs_b_mean_abs"],
    }
    return rows, summary


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Probe whether matched-target conditioning and topology-blend levers actually change the executed model path."
    )
    parser.add_argument("--config", type=Path, required=True, help="Experiment config JSON.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory for CSV/JSON outputs.")
    parser.add_argument("--checkpoint", type=Path, default=None, help="Optional checkpoint to load before probing.")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--input-seed", type=int, default=123)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--style-id", type=int, default=1)
    parser.add_argument("--latent-size", type=int, default=32)
    parser.add_argument(
        "--override",
        action="append",
        default=[],
        help="Repeatable config override in section.field=JSON_VALUE form, e.g. --override model.style_code_spatial_mode=\"lowrank\"",
    )
    args = parser.parse_args()

    cfg = load_experiment_config(args.config)
    applied_overrides = _apply_config_overrides(cfg, list(args.override))
    device = torch.device(args.device)
    inputs = _random_inputs(
        batch_size=args.batch_size,
        latent_channels=int(cfg.model.latent_channels),
        latent_size=args.latent_size,
        style_id=args.style_id,
        seed=args.input_seed,
        device=device,
    )
    conditioning_rows, conditioning_summary = _conditioning_rows(
        cfg,
        device=device,
        seed=args.seed,
        checkpoint=args.checkpoint,
        inputs=inputs,
    )
    topology_rows, topology_pair_rows, topology_summary = _topology_rows(
        cfg,
        device=device,
        seed=args.seed,
        checkpoint=args.checkpoint,
        inputs=inputs,
    )
    anatomy_rows, anatomy_summary = _path_anatomy_rows(
        cfg,
        device=device,
        seed=args.seed,
        checkpoint=args.checkpoint,
        inputs=inputs,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(args.output_dir / "conditioning_sensitivity.csv", conditioning_rows)
    _write_csv(args.output_dir / "topology_sensitivity.csv", topology_rows)
    _write_csv(args.output_dir / "topology_pairwise.csv", topology_pair_rows)
    _write_csv(args.output_dir / "path_anatomy.csv", anatomy_rows)
    (args.output_dir / "effective_config.json").write_text(
        json.dumps(cfg.to_dict(), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    summary = {
        "output_dir": str(args.output_dir),
        "config": str(args.config),
        "checkpoint": str(args.checkpoint) if args.checkpoint else None,
        "applied_overrides": applied_overrides,
        "git_commit": _git_commit(),
        "runtime_metadata": _runtime_metadata(args.device, device),
        "seed": int(args.seed),
        "input_seed": int(args.input_seed),
        "batch_size": int(args.batch_size),
        "style_id": int(args.style_id),
        "latent_size": int(args.latent_size),
        "conditioning_rows": conditioning_rows,
        "topology_rows": topology_rows,
        "topology_pair_rows": topology_pair_rows,
        "path_anatomy_rows": anatomy_rows,
        **conditioning_summary,
        **topology_summary,
        **anatomy_summary,
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(args.output_dir / "summary.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
