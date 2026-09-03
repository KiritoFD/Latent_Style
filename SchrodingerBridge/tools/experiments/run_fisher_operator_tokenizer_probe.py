from __future__ import annotations

"""Fisher-to-operator tokenizer probe.

This is a no-training diagnostic.  Fisher style coordinates are useful only if
they can be consumed by an executable operator.  This script projects internal
style descriptors into a Fisher-discriminative space, anchors those axes to
measured mid/high frequency energy, writes them directly into tokenizer band and
grammar fields, then measures the endpoint motion and transport-AdaIN debug
readouts before any full evaluation is allowed.
"""

import argparse
import copy
import csv
import json
import random
import sys
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from model import build_model_from_config  # noqa: E402
from ot_cost import SWDTransportCost  # noqa: E402
from run_style_embedding_distill import (  # noqa: E402
    _gradient_cosine_loss,
    _integrate_with_grad,
    _load_latent,
    _memory_tier_eval_batch_size,
    _run_full_eval,
    _sample_latent_batch,
    _save_style_adapter,
    _style_latent_index,
    _tv_loss,
)
from run_style_embedding_mainline_calibration import _apply_style_adapter  # noqa: E402
from run_style_memory_bank_adapter_probe import (  # noqa: E402
    _edge_energy,
    _encode_body_features,
    _fisher_project_descriptors,
    _style_measure_descriptor,
)
from run_tokenizer_adain_gate_calibration import _resolve_latent_root, _resolve_path  # noqa: E402


FIELD_SPECS = [
    ("band", 0, "band_low"),
    ("band", 1, "band_mid"),
    ("band", 2, "band_high"),
    ("grammar", 1, "grammar_flatness"),
    ("grammar", 5, "grammar_mid_texton"),
    ("grammar", 6, "grammar_high_texture"),
    ("grammar", 7, "grammar_high_suppression"),
    ("grammar", 8, "grammar_filter_laplace"),
    ("grammar", 9, "grammar_filter_laplace8"),
    ("grammar", 10, "grammar_filter_sobel_x"),
    ("grammar", 11, "grammar_filter_sobel_y"),
    ("grammar", 12, "grammar_filter_diag_a"),
    ("grammar", 13, "grammar_filter_diag_b"),
    ("grammar", 14, "grammar_filter_checker"),
    ("grammar", 15, "grammar_filter_inv_laplace"),
]


def _l2_mean(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return (a.float() - b.float()).square().mean()


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def _lowpass(x: torch.Tensor, kernel: int) -> torch.Tensor:
    kernel = max(1, int(kernel))
    if kernel <= 1:
        return x.float()
    if kernel % 2 == 0:
        kernel += 1
    return F.avg_pool2d(x.float(), kernel_size=kernel, stride=1, padding=kernel // 2)


def _zscore(values: torch.Tensor) -> torch.Tensor:
    values = values.float()
    return (values - values.mean()) / values.std(unbiased=False).clamp_min(1e-6)


def _band_metrics(delta: torch.Tensor) -> dict[str, float]:
    delta_f = delta.float()
    low = _lowpass(delta_f, 9)
    inner = _lowpass(delta_f, 3)
    mid = inner - low
    high = delta_f - inner
    endpoint_rms = torch.sqrt(delta_f.square().mean()).clamp_min(1e-8)
    low_rms = torch.sqrt(low.square().mean())
    mid_rms = torch.sqrt(mid.square().mean())
    high_rms = torch.sqrt(high.square().mean())
    detail_rms = torch.sqrt((mid + high).square().mean())
    return {
        "endpoint_delta_rms": float(endpoint_rms.item()),
        "endpoint_delta_abs_mean": float(delta_f.abs().mean().item()),
        "low_delta_rms": float(low_rms.item()),
        "mid_delta_rms": float(mid_rms.item()),
        "high_delta_rms": float(high_rms.item()),
        "detail_delta_rms": float(detail_rms.item()),
        "low_fraction": float((low_rms / endpoint_rms).item()),
        "mid_fraction": float((mid_rms / endpoint_rms).item()),
        "high_fraction": float((high_rms / endpoint_rms).item()),
        "detail_over_low": float((detail_rms / low_rms.clamp_min(1e-8)).item()),
    }


def _debug_scalar(model: torch.nn.Module, key: str) -> float:
    debug = dict(getattr(model, "carrier_debug", {}) or {})
    value = debug.get(key)
    if torch.is_tensor(value):
        return float(value.detach().float().mean().cpu().item())
    return float("nan")


def _load_model(
    checkpoint: Path,
    *,
    init_style_adapter: Path,
    texture_scale: float,
    band_gain_scale: float,
    flatten_strength: float,
    flatten_kernel: int,
    depthwise_filter_enable: bool,
    depthwise_filter_strength: float,
    depthwise_filter_tanh_scale: float,
    depthwise_filter_basis_offset: int,
    device: str,
) -> tuple[torch.nn.Module, dict]:
    ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
    config = copy.deepcopy(ckpt["config"])
    model_cfg = config.setdefault("model", {})
    model_cfg.update(
        {
            "style_tokenizer_enable": True,
            "style_token_identity_dim": int(model_cfg.get("style_token_identity_dim", 16)),
            "style_token_grammar_dim": max(
                32,
                int(model_cfg.get("style_token_grammar_dim", 32)),
                int(depthwise_filter_basis_offset) + 8,
            ),
            "style_token_band_dim": 3,
            "style_token_code_residual_scale": 1.0,
            "style_token_band_gain_scale": float(band_gain_scale),
            "style_token_learn_identity": False,
            "style_token_flatten_strength": float(flatten_strength),
            "style_token_flatten_kernel": int(flatten_kernel),
            "style_token_adain_gate_enable": True,
            "style_token_reader_enable": False,
            "style_token_grammar_texture_enable": True,
            "style_token_grammar_texture_scale": float(texture_scale),
            "style_token_depthwise_filter_enable": bool(depthwise_filter_enable),
            "style_token_depthwise_filter_strength": float(depthwise_filter_strength),
            "style_token_depthwise_filter_tanh_scale": float(depthwise_filter_tanh_scale),
            "style_token_depthwise_filter_basis_offset": int(depthwise_filter_basis_offset),
        }
    )
    state = ckpt["model_state_dict"]
    if any(k.startswith("_orig_mod.") for k in state):
        state = {k.replace("_orig_mod.", ""): v for k, v in state.items()}
    model = build_model_from_config(config["model"], use_checkpointing=False).to(device)
    missing, unexpected = model.load_state_dict(state, strict=False)
    unexpected_clean = [key for key in unexpected if not key.startswith("style_tokenizer.")]
    if unexpected_clean:
        raise RuntimeError(f"Unexpected non-tokenizer checkpoint keys: {unexpected_clean[:8]}")
    _apply_style_adapter(model, init_style_adapter, device)
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)
    model._tokenizer_load_missing = list(missing)
    model._tokenizer_load_unexpected = list(unexpected)
    return model, config


def _physical_style_stats(feats: torch.Tensor) -> dict[str, torch.Tensor]:
    feats_f = feats.float()
    low = _lowpass(feats_f, 9)
    inner = _lowpass(feats_f, 3)
    mid = inner - low
    high = feats_f - inner
    return {
        "low_std": low.std(dim=(1, 2, 3), unbiased=False),
        "mid_abs": mid.abs().mean(dim=(1, 2, 3)),
        "high_abs": high.abs().mean(dim=(1, 2, 3)),
        "edge": _edge_energy(feats_f, kernel=5),
    }


def _pick_aligned_axis(coords: torch.Tensor, target: torch.Tensor, used: set[int]) -> torch.Tensor:
    if coords.ndim != 2 or coords.shape[1] == 0:
        return torch.zeros_like(target)
    target = _zscore(target)
    best_idx = 0
    best_score = -1.0
    best_sign = 1.0
    for idx in range(coords.shape[1]):
        if idx in used:
            continue
        axis = _zscore(coords[:, idx])
        corr = float((axis * target).mean().item())
        score = abs(corr)
        if score > best_score:
            best_idx = idx
            best_score = score
            best_sign = 1.0 if corr >= 0 else -1.0
    used.add(best_idx)
    return _zscore(coords[:, best_idx]) * best_sign


@torch.inference_mode()
def _build_fisher_operator_tokens(
    model: torch.nn.Module,
    latent_index: dict[str, list[Path]],
    *,
    style_names: list[str],
    sample_count: int,
    batch_size: int,
    descriptor_kernel: int,
    fisher_dim: int,
    fisher_reg: float,
    band_scale: float,
    grammar_scale: float,
    flat_scale: float,
    clamp: float,
    device: str,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor, list[dict[str, Any]]]:
    rng = random.Random(seed)
    descriptors_by_style: list[torch.Tensor] = []
    stat_rows: list[dict[str, Any]] = []
    low_stats: list[torch.Tensor] = []
    mid_stats: list[torch.Tensor] = []
    high_stats: list[torch.Tensor] = []
    edge_stats: list[torch.Tensor] = []

    for style_id, style_name in enumerate(style_names):
        paths = list(latent_index[style_name])
        rng.shuffle(paths)
        paths = paths[: min(max(1, int(sample_count)), len(paths))]
        chunks: list[torch.Tensor] = []
        for start in range(0, len(paths), max(1, int(batch_size))):
            batch_paths = paths[start : start + max(1, int(batch_size))]
            latents = torch.cat([_load_latent(path) for path in batch_paths], dim=0)
            chunks.append(_encode_body_features(model, latents, style_id, device).cpu())
        feats = torch.cat(chunks, dim=0)
        descriptors_by_style.append(_style_measure_descriptor(feats, descriptor_kernel).cpu())
        stats = _physical_style_stats(feats)
        low_stats.append(stats["low_std"].mean())
        mid_stats.append(stats["mid_abs"].mean())
        high_stats.append(stats["high_abs"].mean())
        edge_stats.append(stats["edge"].mean())
        stat_rows.append(
            {
                "style_id": style_id,
                "style_name": style_name,
                "sample_count": int(feats.shape[0]),
                "low_std": float(low_stats[-1].item()),
                "mid_abs": float(mid_stats[-1].item()),
                "high_abs": float(high_stats[-1].item()),
                "edge": float(edge_stats[-1].item()),
            }
        )

    fisher = _fisher_project_descriptors(
        descriptors_by_style,
        max_dim=max(1, int(fisher_dim)),
        reg=float(fisher_reg),
    )
    centroids = torch.stack([desc.mean(dim=0) for desc in fisher], dim=0)
    low_z = _zscore(torch.stack(low_stats))
    mid_z = _zscore(torch.stack(mid_stats))
    high_z = _zscore(torch.stack(high_stats))
    edge_z = _zscore(torch.stack(edge_stats))
    used: set[int] = set()
    fisher_mid = _pick_aligned_axis(centroids, mid_z + 0.35 * edge_z, used)
    fisher_high = _pick_aligned_axis(centroids, high_z, used)
    fisher_flat = -high_z

    band = torch.zeros((len(style_names), 3), dtype=torch.float32)
    grammar = torch.zeros((len(style_names), 32), dtype=torch.float32)
    band[:, 0] = (low_z - 0.20 * high_z) * float(band_scale)
    band[:, 1] = (mid_z + 0.40 * fisher_mid) * float(band_scale)
    band[:, 2] = (high_z + 0.40 * fisher_high) * float(band_scale)
    grammar[:, 1] = fisher_flat * float(flat_scale)
    grammar[:, 5] = (fisher_mid + 0.50 * mid_z) * float(grammar_scale)
    grammar[:, 6] = (fisher_high + 0.50 * high_z) * float(grammar_scale)
    grammar[:, 7] = fisher_flat * float(flat_scale)
    filter_scale = float(grammar_scale) * 0.80
    grammar[:, 8] = fisher_mid * filter_scale
    grammar[:, 9] = fisher_high * filter_scale
    grammar[:, 10] = edge_z * filter_scale
    grammar[:, 11] = mid_z * filter_scale
    grammar[:, 12] = (0.60 * fisher_mid + 0.40 * edge_z) * filter_scale
    grammar[:, 13] = (0.60 * fisher_high + 0.40 * high_z) * filter_scale
    grammar[:, 14] = (edge_z - high_z) * filter_scale
    grammar[:, 15] = fisher_flat * filter_scale

    band = band.clamp(-float(clamp), float(clamp))
    grammar = grammar.clamp(-float(clamp), float(clamp))
    # Keep the photo identity neutral.  It is source-domain control, not a style target.
    band[0].zero_()
    grammar[0].zero_()

    for idx, row in enumerate(stat_rows):
        row.update(
            {
                "low_z": float(low_z[idx].item()),
                "mid_z": float(mid_z[idx].item()),
                "high_z": float(high_z[idx].item()),
                "edge_z": float(edge_z[idx].item()),
                "fisher_mid": float(fisher_mid[idx].item()),
                "fisher_high": float(fisher_high[idx].item()),
                "fisher_flat": float(fisher_flat[idx].item()),
                "band_low": float(band[idx, 0].item()),
                "band_mid": float(band[idx, 1].item()),
                "band_high": float(band[idx, 2].item()),
                "grammar_flatness": float(grammar[idx, 1].item()),
                "grammar_mid_texton": float(grammar[idx, 5].item()),
                "grammar_high_texture": float(grammar[idx, 6].item()),
                "grammar_high_suppression": float(grammar[idx, 7].item()),
                "grammar_filter_laplace": float(grammar[idx, 8].item()),
                "grammar_filter_laplace8": float(grammar[idx, 9].item()),
                "grammar_filter_sobel_x": float(grammar[idx, 10].item()),
                "grammar_filter_sobel_y": float(grammar[idx, 11].item()),
                "grammar_filter_diag_a": float(grammar[idx, 12].item()),
                "grammar_filter_diag_b": float(grammar[idx, 13].item()),
                "grammar_filter_checker": float(grammar[idx, 14].item()),
                "grammar_filter_inv_laplace": float(grammar[idx, 15].item()),
            }
        )
    return grammar, band, stat_rows


def _apply_tokens(model: torch.nn.Module, grammar: torch.Tensor, band: torch.Tensor) -> None:
    tokenizer = getattr(model, "style_tokenizer", None)
    if tokenizer is None:
        raise RuntimeError("tokenizer was not constructed")
    with torch.no_grad():
        tokenizer.grammar_vocab.weight.zero_()
        tokenizer.band_vocab.weight.zero_()
        g_cols = min(grammar.shape[1], tokenizer.grammar_vocab.weight.shape[1])
        b_cols = min(band.shape[1], tokenizer.band_vocab.weight.shape[1])
        tokenizer.grammar_vocab.weight[:, :g_cols].copy_(
            grammar[:, :g_cols].to(device=tokenizer.grammar_vocab.weight.device, dtype=tokenizer.grammar_vocab.weight.dtype)
        )
        tokenizer.band_vocab.weight[:, :b_cols].copy_(
            band[:, :b_cols].to(device=tokenizer.band_vocab.weight.device, dtype=tokenizer.band_vocab.weight.dtype)
        )


@torch.no_grad()
def _endpoint(model: torch.nn.Module, content: torch.Tensor, style_id: int, ode_steps: int) -> torch.Tensor:
    sid = torch.full((content.shape[0],), int(style_id), dtype=torch.long, device=content.device)
    return _integrate_with_grad(model, content, style_id=sid, num_steps=int(ode_steps))


def _perturb_token(model: torch.nn.Module, field: str, dim: int, style_id: int, delta: float) -> torch.Tensor:
    tokenizer = getattr(model, "style_tokenizer", None)
    if tokenizer is None:
        raise RuntimeError("tokenizer was not constructed")
    table = tokenizer.band_vocab.weight if field == "band" else tokenizer.grammar_vocab.weight
    with torch.no_grad():
        original = table[int(style_id), int(dim)].detach().clone()
        table[int(style_id), int(dim)] = original + float(delta)
    return original


def _restore_token(model: torch.nn.Module, field: str, dim: int, style_id: int, value: torch.Tensor) -> None:
    tokenizer = getattr(model, "style_tokenizer", None)
    if tokenizer is None:
        raise RuntimeError("tokenizer was not constructed")
    table = tokenizer.band_vocab.weight if field == "band" else tokenizer.grammar_vocab.weight
    with torch.no_grad():
        table[int(style_id), int(dim)] = value.to(device=table.device, dtype=table.dtype)


def _mean_rows(rows: list[dict[str, Any]], *, keys: list[str]) -> dict[str, float]:
    out: dict[str, float] = {}
    for key in keys:
        values = []
        for row in rows:
            try:
                values.append(float(row[key]))
            except (KeyError, TypeError, ValueError):
                pass
        out[key] = sum(values) / max(1, len(values))
    return out


def _run_readout(args: argparse.Namespace) -> None:
    checkpoint = _resolve_path(args.checkpoint)
    init_style_adapter = _resolve_path(args.init_style_adapter)
    if checkpoint is None or init_style_adapter is None:
        raise ValueError("checkpoint and init-style-adapter are required")
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    latent_root = _resolve_latent_root(ckpt["config"], args.latent_root)
    style_names = [item.strip() for item in args.style_subdirs.split(",") if item.strip()]
    target_style_ids = [int(item.strip()) for item in args.target_style_ids.split(",") if item.strip()]

    neutral, config = _load_model(
        checkpoint,
        init_style_adapter=init_style_adapter,
        texture_scale=float(args.texture_scale),
        band_gain_scale=float(args.band_gain_scale),
        flatten_strength=float(args.flatten_strength),
        flatten_kernel=int(args.flatten_kernel),
        depthwise_filter_enable=bool(args.depthwise_filter_enable),
        depthwise_filter_strength=float(args.depthwise_filter_strength),
        depthwise_filter_tanh_scale=float(args.depthwise_filter_tanh_scale),
        depthwise_filter_basis_offset=int(args.depthwise_filter_basis_offset),
        device=args.device,
    )
    fisher_model, _ = _load_model(
        checkpoint,
        init_style_adapter=init_style_adapter,
        texture_scale=float(args.texture_scale),
        band_gain_scale=float(args.band_gain_scale),
        flatten_strength=float(args.flatten_strength),
        flatten_kernel=int(args.flatten_kernel),
        depthwise_filter_enable=bool(args.depthwise_filter_enable),
        depthwise_filter_strength=float(args.depthwise_filter_strength),
        depthwise_filter_tanh_scale=float(args.depthwise_filter_tanh_scale),
        depthwise_filter_basis_offset=int(args.depthwise_filter_basis_offset),
        device=args.device,
    )
    latent_index = _style_latent_index(latent_root, style_names)
    grammar, band, token_rows = _build_fisher_operator_tokens(
        fisher_model,
        latent_index,
        style_names=style_names,
        sample_count=int(args.sample_count),
        batch_size=int(args.encode_batch_size),
        descriptor_kernel=int(args.descriptor_kernel),
        fisher_dim=int(args.fisher_dim),
        fisher_reg=float(args.fisher_reg),
        band_scale=float(args.token_band_scale),
        grammar_scale=float(args.token_grammar_scale),
        flat_scale=float(args.token_flat_scale),
        clamp=float(args.token_clamp),
        device=args.device,
        seed=int(args.seed),
    )
    _apply_tokens(fisher_model, grammar, band)

    rng = random.Random(int(args.seed) + 17)
    content_pool = [p for style in style_names for p in latent_index[style]]
    preview_rows: list[dict[str, Any]] = []
    perturb_rows: list[dict[str, Any]] = []

    for style_id in target_style_ids:
        style_name = style_names[style_id]
        for batch_idx in range(1, max(1, int(args.num_batches)) + 1):
            content = _sample_latent_batch(content_pool, int(args.batch_size), args.device, rng)
            base = _endpoint(neutral, content, style_id, int(args.ode_steps))
            fisher = _endpoint(fisher_model, content, style_id, int(args.ode_steps))
            row: dict[str, Any] = {
                "kind": "fisher_token_preview",
                "style_id": style_id,
                "style": style_name,
                "batch": batch_idx,
                "grammar_mid_alloc_mean": _debug_scalar(fisher_model, "body_transport_adain_grammar_mid_alloc"),
                "grammar_high_alloc_mean": _debug_scalar(fisher_model, "body_transport_adain_grammar_high_alloc"),
                "band_alloc_mean": _debug_scalar(fisher_model, "body_transport_adain_band_alloc"),
                "band_low_token": float(band[style_id, 0].item()),
                "band_mid_token": float(band[style_id, 1].item()),
                "band_high_token": float(band[style_id, 2].item()),
                "grammar_mid_token": float(grammar[style_id, 5].item()),
                "grammar_high_token": float(grammar[style_id, 6].item()),
            }
            row.update(_band_metrics(fisher - base))
            preview_rows.append(row)

            for field, dim, label in FIELD_SPECS:
                for sign in (-1.0, 1.0):
                    original = _perturb_token(fisher_model, field, dim, style_id, sign * float(args.perturb_delta))
                    try:
                        perturbed = _endpoint(fisher_model, content, style_id, int(args.ode_steps))
                    finally:
                        _restore_token(fisher_model, field, dim, style_id, original)
                    prow: dict[str, Any] = {
                        "kind": "local_perturbation",
                        "style_id": style_id,
                        "style": style_name,
                        "batch": batch_idx,
                        "field": field,
                        "dim": dim,
                        "label": label,
                        "delta": sign * float(args.perturb_delta),
                        "grammar_mid_alloc_mean": _debug_scalar(fisher_model, "body_transport_adain_grammar_mid_alloc"),
                        "grammar_high_alloc_mean": _debug_scalar(fisher_model, "body_transport_adain_grammar_high_alloc"),
                        "band_alloc_mean": _debug_scalar(fisher_model, "body_transport_adain_band_alloc"),
                    }
                    prow.update(_band_metrics(perturbed - fisher))
                    perturb_rows.append(prow)
            del content, base, fisher
            if str(args.device).startswith("cuda"):
                torch.cuda.empty_cache()

    metric_keys = [
        "endpoint_delta_rms",
        "low_delta_rms",
        "mid_delta_rms",
        "high_delta_rms",
        "detail_over_low",
        "high_fraction",
    ]
    summary_rows: list[dict[str, Any]] = []
    for style_id in target_style_ids:
        style_rows = [row for row in preview_rows if int(row["style_id"]) == style_id]
        summary_rows.append(
            {
                "group": "fisher_token_preview",
                "style_id": style_id,
                "style": style_names[style_id],
                **_mean_rows(style_rows, keys=metric_keys),
            }
        )
    for _, _, label in FIELD_SPECS:
        field_rows = [row for row in perturb_rows if row["label"] == label]
        summary_rows.append({"group": f"perturb_{label}", "style_id": "", "style": "", **_mean_rows(field_rows, keys=metric_keys)})

    preview_summary = _mean_rows(preview_rows, keys=metric_keys)
    promoted = (
        preview_summary["endpoint_delta_rms"] > float(args.promote_min_endpoint_rms)
        and preview_summary["detail_over_low"] > float(args.promote_min_detail_over_low)
        and preview_summary["high_fraction"] > float(args.promote_min_high_fraction)
    )
    decision = (
        "promote_to_tokenizer_training"
        if promoted
        else "reject_or_rethink_operator_mapping_before_training"
    )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(args.out_dir / "fisher_operator_token_rows.csv", token_rows)
    _write_csv(args.out_dir / "fisher_operator_preview_rows.csv", preview_rows)
    _write_csv(args.out_dir / "fisher_operator_perturb_rows.csv", perturb_rows)
    _write_csv(args.out_dir / "fisher_operator_summary.csv", summary_rows)
    _write_json(
        args.out_dir / "manifest.json",
        {
            "checkpoint": str(checkpoint),
            "init_style_adapter": str(init_style_adapter),
            "latent_root": str(latent_root),
            "style_names": style_names,
            "target_style_ids": target_style_ids,
            "config_model_overrides": {
                "style_tokenizer_enable": True,
                "style_token_band_gain_scale": float(args.band_gain_scale),
                "style_token_grammar_texture_scale": float(args.texture_scale),
                "style_token_flatten_strength": float(args.flatten_strength),
                "style_token_depthwise_filter_enable": bool(args.depthwise_filter_enable),
                "style_token_depthwise_filter_strength": float(args.depthwise_filter_strength),
                "style_token_depthwise_filter_tanh_scale": float(args.depthwise_filter_tanh_scale),
            },
            "token_mapping": {
                "band0": "low_z - 0.20 high_z",
                "band1": "mid_z + 0.40 FisherAxis(mid/edge)",
                "band2": "high_z + 0.40 FisherAxis(high)",
                "grammar1": "negative high_z flatness",
                "grammar5": "FisherAxis(mid/edge) + 0.50 mid_z",
                "grammar6": "FisherAxis(high) + 0.50 high_z",
                "grammar7": "negative high_z suppression",
                "grammar8_15": "Fisher/physical coordinates for fixed depthwise 3x3 filter basis",
            },
            "promotion_gate": {
                "min_endpoint_rms": float(args.promote_min_endpoint_rms),
                "min_detail_over_low": float(args.promote_min_detail_over_low),
                "min_high_fraction": float(args.promote_min_high_fraction),
                "observed": preview_summary,
                "decision": decision,
            },
            "missing_keys_from_source": getattr(fisher_model, "_tokenizer_load_missing", []),
            "unexpected_keys_from_source": getattr(fisher_model, "_tokenizer_load_unexpected", []),
            "main_omf_loss_changed": False,
        },
    )

    top = sorted(preview_rows + perturb_rows, key=lambda item: float(item.get("endpoint_delta_rms", 0.0)), reverse=True)[:20]
    lines = [
        "# Fisher Operator Tokenizer Probe",
        "",
        "No training. Main OMF loss unchanged.",
        "",
        "One-line hypothesis: Fisher style axes become useful only when sign/order are anchored by measured mid/high energy and written into executable transport-AdaIN tokenizer fields.",
        "",
        f"Decision: `{decision}`",
        "",
        "## Preview Summary",
        "",
        "| metric | value |",
        "|---|---:|",
    ]
    for key, value in preview_summary.items():
        lines.append(f"| {key} | {value:.6f} |")
    lines += [
        "",
        "## Strongest Local Responses",
        "",
        "| kind | style | label | delta | endpoint | low | mid | high | detail/low | high_frac |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in top:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row.get("kind", "")),
                    str(row.get("style", "")),
                    str(row.get("label", "fisher_token")),
                    f"{float(row.get('delta', 0.0)):.3f}" if row.get("kind") == "local_perturbation" else "",
                    f"{float(row.get('endpoint_delta_rms', 0.0)):.6f}",
                    f"{float(row.get('low_delta_rms', 0.0)):.6f}",
                    f"{float(row.get('mid_delta_rms', 0.0)):.6f}",
                    f"{float(row.get('high_delta_rms', 0.0)):.6f}",
                    f"{float(row.get('detail_over_low', 0.0)):.3f}",
                    f"{float(row.get('high_fraction', 0.0)):.3f}",
                ]
            )
            + " |"
        )
    lines += [
        "",
        "## Interpretation Gate",
        "",
        "- Promote only if the preview motion is not mostly low/color drift.",
        "- If grammar perturbations are active but preview is weak, the field is executable but the Fisher mapping is wrong.",
        "- If both preview and perturbations are weak, do not train; change the operator binding first.",
    ]
    (args.out_dir / "fisher_operator_readout.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(args.out_dir)


def _save_checkpoint(
    path: Path,
    model: torch.nn.Module,
    config: dict,
    *,
    source_checkpoint: Path,
    init_style_adapter: Path,
    train_args: argparse.Namespace,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": {key: value.detach().cpu() for key, value in model.state_dict().items()},
            "config": config,
            "fisher_operator_tokenizer_source_checkpoint": str(source_checkpoint),
            "fisher_operator_tokenizer_init_style_adapter": str(init_style_adapter),
            "fisher_operator_tokenizer_train_args": {
                "train_recipe_name": str(train_args.train_recipe_name),
                "iters_per_style": int(train_args.iters_per_style),
                "train_batch_size": int(train_args.train_batch_size),
                "lr": float(train_args.lr),
                "swd_weight": float(train_args.swd_weight),
                "hp_swd_weight": float(train_args.hp_swd_weight),
                "anchor_weight": float(train_args.anchor_weight),
                "grad_weight": float(train_args.grad_weight),
                "delta_tv_weight": float(train_args.delta_tv_weight),
                "token_l2_weight": float(train_args.token_l2_weight),
                "depthwise_filter_enable": bool(train_args.depthwise_filter_enable),
                "depthwise_filter_strength": float(train_args.depthwise_filter_strength),
                "depthwise_filter_tanh_scale": float(train_args.depthwise_filter_tanh_scale),
            },
        },
        path,
    )


def _run_train(args: argparse.Namespace) -> None:
    checkpoint = _resolve_path(args.checkpoint)
    init_style_adapter = _resolve_path(args.init_style_adapter)
    if checkpoint is None or init_style_adapter is None:
        raise ValueError("checkpoint and init-style-adapter are required")
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    latent_root = _resolve_latent_root(ckpt["config"], args.latent_root)
    style_names = [item.strip() for item in args.style_subdirs.split(",") if item.strip()]
    target_style_ids = [int(item.strip()) for item in args.target_style_ids.split(",") if item.strip()]
    rng = random.Random(int(args.seed) + 313)

    model, config = _load_model(
        checkpoint,
        init_style_adapter=init_style_adapter,
        texture_scale=float(args.texture_scale),
        band_gain_scale=float(args.band_gain_scale),
        flatten_strength=float(args.flatten_strength),
        flatten_kernel=int(args.flatten_kernel),
        depthwise_filter_enable=bool(args.depthwise_filter_enable),
        depthwise_filter_strength=float(args.depthwise_filter_strength),
        depthwise_filter_tanh_scale=float(args.depthwise_filter_tanh_scale),
        depthwise_filter_basis_offset=int(args.depthwise_filter_basis_offset),
        device=args.device,
    )
    teacher, _ = _load_model(
        checkpoint,
        init_style_adapter=init_style_adapter,
        texture_scale=float(args.texture_scale),
        band_gain_scale=float(args.band_gain_scale),
        flatten_strength=float(args.flatten_strength),
        flatten_kernel=int(args.flatten_kernel),
        depthwise_filter_enable=bool(args.depthwise_filter_enable),
        depthwise_filter_strength=float(args.depthwise_filter_strength),
        depthwise_filter_tanh_scale=float(args.depthwise_filter_tanh_scale),
        depthwise_filter_basis_offset=int(args.depthwise_filter_basis_offset),
        device=args.device,
    )
    teacher.eval()
    for param in teacher.parameters():
        param.requires_grad_(False)

    latent_index = _style_latent_index(latent_root, style_names)
    grammar, band, token_rows = _build_fisher_operator_tokens(
        model,
        latent_index,
        style_names=style_names,
        sample_count=int(args.sample_count),
        batch_size=int(args.encode_batch_size),
        descriptor_kernel=int(args.descriptor_kernel),
        fisher_dim=int(args.fisher_dim),
        fisher_reg=float(args.fisher_reg),
        band_scale=float(args.token_band_scale),
        grammar_scale=float(args.token_grammar_scale),
        flat_scale=float(args.token_flat_scale),
        clamp=float(args.token_clamp),
        device=args.device,
        seed=int(args.seed),
    )
    _apply_tokens(model, grammar, band)

    model.train()
    for param in model.parameters():
        param.requires_grad_(False)
    tokenizer = getattr(model, "style_tokenizer", None)
    if tokenizer is None:
        raise RuntimeError("tokenizer was not constructed")
    tokenizer.grammar_vocab.weight.requires_grad_(True)
    tokenizer.band_vocab.weight.requires_grad_(True)
    params = [tokenizer.grammar_vocab.weight, tokenizer.band_vocab.weight]
    base_grammar = tokenizer.grammar_vocab.weight.detach().clone()
    base_band = tokenizer.band_vocab.weight.detach().clone()
    optimizer = torch.optim.AdamW(params, lr=float(args.lr), weight_decay=0.0)
    transport = SWDTransportCost(config)
    content_pool = [p for style in style_names for p in latent_index[style]]

    recipe_dir = args.out_root / str(args.train_recipe_name)
    recipe_dir.mkdir(parents=True, exist_ok=True)
    loss_rows: list[dict[str, Any]] = []
    start_time = time.time()
    iters_per_style = max(1, int(args.iters_per_style))
    train_batch_size = max(1, int(args.train_batch_size))

    for style_id in target_style_ids:
        style_name = style_names[style_id]
        for iteration in range(1, iters_per_style + 1):
            content = _sample_latent_batch(content_pool, train_batch_size, args.device, rng)
            target = _sample_latent_batch(latent_index[style_name], train_batch_size, args.device, rng)
            sid = torch.full((train_batch_size,), int(style_id), dtype=torch.long, device=args.device)
            optimizer.zero_grad(set_to_none=True)
            pred = _integrate_with_grad(model, content, style_id=sid, num_steps=int(args.ode_steps))
            with torch.no_grad():
                teacher_pred = _integrate_with_grad(teacher, content, style_id=sid, num_steps=int(args.ode_steps))
            swd = transport.aligned_cost(pred, target)
            hp_pred = pred.float() - _lowpass(pred, int(args.highpass_kernel))
            hp_target = target.float() - _lowpass(target, int(args.highpass_kernel))
            hp_swd = transport.aligned_cost(hp_pred, hp_target)
            anchor = _l2_mean(pred, teacher_pred)
            grad = _gradient_cosine_loss(pred, content) if float(args.grad_weight) > 0.0 else pred.new_tensor(0.0)
            tv = _tv_loss(pred - content) if float(args.delta_tv_weight) > 0.0 else pred.new_tensor(0.0)
            token_l2 = _l2_mean(tokenizer.grammar_vocab.weight, base_grammar) + _l2_mean(tokenizer.band_vocab.weight, base_band)
            loss = (
                float(args.swd_weight) * swd
                + float(args.hp_swd_weight) * hp_swd
                + float(args.anchor_weight) * anchor
                + float(args.grad_weight) * grad
                + float(args.delta_tv_weight) * tv
                + float(args.token_l2_weight) * token_l2
            )
            if not torch.isfinite(loss.detach()):
                raise FloatingPointError(f"Non-finite loss style={style_name} iter={iteration}")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, float(args.grad_clip))
            optimizer.step()

            with torch.no_grad():
                gains = 1.0 + torch.tanh(tokenizer.band_vocab.weight[:, :3]) * float(args.band_gain_scale)
                debug = dict(getattr(model, "carrier_debug", {}) or {})
                row = {
                    "recipe": str(args.train_recipe_name),
                    "style_id": int(style_id),
                    "style_name": style_name,
                    "iter": iteration,
                    "loss": float(loss.detach().item()),
                    "swd": float(swd.detach().item()),
                    "hp_swd": float(hp_swd.detach().item()),
                    "anchor": float(anchor.detach().item()),
                    "grad": float(grad.detach().item()),
                    "tv": float(tv.detach().item()),
                    "token_l2": float(token_l2.detach().item()),
                    "gain_low": float(gains[style_id, 0].detach().item()),
                    "gain_mid": float(gains[style_id, 1].detach().item()),
                    "gain_high": float(gains[style_id, 2].detach().item()),
                    "grammar_mid": float(tokenizer.grammar_vocab.weight[style_id, 5].detach().item()),
                    "grammar_high": float(tokenizer.grammar_vocab.weight[style_id, 6].detach().item()),
                    "debug_mid_alloc": float(debug.get("body_transport_adain_grammar_mid_alloc", torch.tensor(float("nan"))).detach().float().mean().item())
                    if torch.is_tensor(debug.get("body_transport_adain_grammar_mid_alloc"))
                    else float("nan"),
                    "debug_high_alloc": float(debug.get("body_transport_adain_grammar_high_alloc", torch.tensor(float("nan"))).detach().float().mean().item())
                    if torch.is_tensor(debug.get("body_transport_adain_grammar_high_alloc"))
                    else float("nan"),
                    "depthwise_filter_rms": float(
                        debug.get("body_transport_adain_depthwise_filter_delta", torch.tensor(0.0))
                        .detach()
                        .float()
                        .square()
                        .mean()
                        .sqrt()
                        .item()
                    )
                    if torch.is_tensor(debug.get("body_transport_adain_depthwise_filter_delta"))
                    else 0.0,
                }
                loss_rows.append(row)
            if iteration == 1 or iteration % int(args.print_every) == 0 or iteration == iters_per_style:
                print(
                    f"[{args.train_recipe_name}] style={style_name} iter={iteration}/{iters_per_style} "
                    f"loss={row['loss']:.4f} swd={row['swd']:.4f} hp={row['hp_swd']:.4f} "
                    f"anchor={row['anchor']:.5f} gains=({row['gain_low']:.3f},{row['gain_mid']:.3f},{row['gain_high']:.3f}) "
                    f"grammar=({row['grammar_mid']:.3f},{row['grammar_high']:.3f}) dw={row['depthwise_filter_rms']:.5f}"
                )
            del content, target, sid, pred, teacher_pred, loss
            if str(args.device).startswith("cuda"):
                torch.cuda.empty_cache()

    adapter_path = recipe_dir / "style_adapter.pt"
    checkpoint_path = recipe_dir / "checkpoint_fisher_operator_tokenizer.pt"
    _save_style_adapter(adapter_path, model)
    _save_checkpoint(
        checkpoint_path,
        model,
        config,
        source_checkpoint=checkpoint,
        init_style_adapter=init_style_adapter,
        train_args=args,
    )
    _write_csv(recipe_dir / "fisher_operator_token_rows.csv", token_rows)
    _write_csv(recipe_dir / "training_losses.csv", loss_rows)
    _write_json(
        recipe_dir / "training_manifest.json",
        {
            "checkpoint": str(checkpoint),
            "init_style_adapter": str(init_style_adapter),
            "latent_root": str(latent_root),
            "style_names": style_names,
            "target_style_ids": target_style_ids,
            "elapsed_seconds": time.time() - start_time,
            "hypothesis": (
                "Train only Fisher-initialized tokenizer band/grammar fields while freezing the m02 transport-AdaIN "
                "backbone and style adapter. This tests whether executable token fields can convert discriminative "
                "style coordinates into endpoint style gain without returning to hazy residual-bank behavior."
            ),
            "main_omf_loss_changed": False,
            "trainable": ["style_tokenizer.grammar_vocab.weight", "style_tokenizer.band_vocab.weight"],
            "operator_binding": "grammar[8:16] controls a fixed depthwise 3x3 filter bank on transport-AdaIN residual detail",
        },
    )

    result: dict[str, Any] = {
        "recipe": str(args.train_recipe_name),
        "adapter_path": str(adapter_path),
        "checkpoint": str(checkpoint_path),
    }
    if not args.skip_eval:
        eval_batch_size = _memory_tier_eval_batch_size(args.device, args.eval_batch_size if args.eval_batch_size > 0 else None)
        full_eval_dir = recipe_dir / "full_eval"
        summary = _run_full_eval(
            checkpoint=checkpoint_path,
            style_adapter=adapter_path,
            output_dir=full_eval_dir,
            batch_size=eval_batch_size,
            vae_model=str(args.vae_model),
        )
        _write_json(recipe_dir / "full_eval_summary.json", summary)
        overview = dict(summary.get("analysis", {}).get("all_pairs_overview", {}) or {})
        hayao = dict((summary.get("analysis", {}).get("cross_by_target_style", {}) or {}).get("Hayao", {}) or {})
        result.update(
            {
                "full_eval_dir": str(full_eval_dir),
                "clip_style": overview.get("clip_style", float("nan")),
                "clip_content": overview.get("clip_content", float("nan")),
                "content_lpips": overview.get("content_lpips", float("nan")),
                "ec": overview.get("edge_consistency", overview.get("ec", float("nan"))),
                "hayao_cross_clip_style": hayao.get("clip_style", float("nan")),
                "hayao_cross_content_lpips": hayao.get("content_lpips", float("nan")),
            }
        )
    _write_csv(args.out_root / "fisher_operator_tokenizer_results.csv", [result])
    print(args.out_root / "fisher_operator_tokenizer_results.csv")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["readout", "train"], default="readout")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--init-style-adapter", type=Path, required=True)
    parser.add_argument("--latent-root", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, default=ROOT / "exp/diagnostics/fisher_operator_tokenizer_probe")
    parser.add_argument("--out-root", type=Path, default=ROOT / "exp/fisher_operator_tokenizer_probe")
    parser.add_argument("--train-recipe-name", type=str, default="fo01_fisher_operator_token_swd80")
    parser.add_argument("--style-subdirs", type=str, default="photo,Hayao,monet,vangogh,cezanne")
    parser.add_argument("--target-style-ids", type=str, default="1,2,3,4")
    parser.add_argument("--sample-count", type=int, default=96)
    parser.add_argument("--encode-batch-size", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=3)
    parser.add_argument("--train-batch-size", type=int, default=14)
    parser.add_argument("--num-batches", type=int, default=2)
    parser.add_argument("--iters-per-style", type=int, default=80)
    parser.add_argument("--ode-steps", type=int, default=12)
    parser.add_argument("--descriptor-kernel", type=int, default=5)
    parser.add_argument("--fisher-dim", type=int, default=4)
    parser.add_argument("--fisher-reg", type=float, default=0.05)
    parser.add_argument("--band-gain-scale", type=float, default=0.24)
    parser.add_argument("--texture-scale", type=float, default=0.42)
    parser.add_argument("--flatten-strength", type=float, default=0.025)
    parser.add_argument("--flatten-kernel", type=int, default=7)
    parser.add_argument("--depthwise-filter-enable", action="store_true")
    parser.add_argument("--depthwise-filter-strength", type=float, default=0.14)
    parser.add_argument("--depthwise-filter-tanh-scale", type=float, default=0.35)
    parser.add_argument("--depthwise-filter-basis-offset", type=int, default=8)
    parser.add_argument("--token-band-scale", type=float, default=1.10)
    parser.add_argument("--token-grammar-scale", type=float, default=1.25)
    parser.add_argument("--token-flat-scale", type=float, default=0.85)
    parser.add_argument("--token-clamp", type=float, default=1.75)
    parser.add_argument("--perturb-delta", type=float, default=0.75)
    parser.add_argument("--lr", type=float, default=1.1e-3)
    parser.add_argument("--swd-weight", type=float, default=0.70)
    parser.add_argument("--hp-swd-weight", type=float, default=1.05)
    parser.add_argument("--anchor-weight", type=float, default=0.16)
    parser.add_argument("--grad-weight", type=float, default=0.12)
    parser.add_argument("--delta-tv-weight", type=float, default=0.04)
    parser.add_argument("--token-l2-weight", type=float, default=0.012)
    parser.add_argument("--highpass-kernel", type=int, default=5)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--print-every", type=int, default=20)
    parser.add_argument("--skip-eval", action="store_true")
    parser.add_argument("--eval-batch-size", type=int, default=0)
    parser.add_argument("--vae-model", type=str, default="auto")
    parser.add_argument("--promote-min-endpoint-rms", type=float, default=1.0e-4)
    parser.add_argument("--promote-min-detail-over-low", type=float, default=0.55)
    parser.add_argument("--promote-min-high-fraction", type=float, default=0.20)
    parser.add_argument("--seed", type=int, default=9751)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    if args.mode == "train":
        _run_train(args)
    else:
        _run_readout(args)


if __name__ == "__main__":
    main()
