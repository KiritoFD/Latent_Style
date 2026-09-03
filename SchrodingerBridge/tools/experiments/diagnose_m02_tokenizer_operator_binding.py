from __future__ import annotations

"""No-training diagnostic for tokenizer operator binding on the m02 anchor.

Question: if grammar[5]/grammar[6] are hard-bound to transport-AdaIN mid/high
residual gains, do measured tokenizer fields finally produce executable
mid/high endpoint motion without changing the main OMF loss?
"""

import argparse
import copy
import csv
import json
import random
import sys
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
from run_style_embedding_distill import (  # noqa: E402
    _integrate_with_grad,
    _sample_latent_batch,
    _style_latent_index,
)
from run_style_embedding_mainline_calibration import _apply_style_adapter  # noqa: E402
from run_tokenizer_adain_gate_calibration import _resolve_latent_root, _resolve_path  # noqa: E402
from run_tokenizer_stat_vocab_probe import (  # noqa: E402
    StatProbeRecipe,
    _build_vocab,
    _sample_style_tensor,
    _style_stats,
)


FIELD_SPECS = [
    ("band", 0, "band_low"),
    ("band", 1, "band_mid"),
    ("band", 2, "band_high"),
    ("grammar", 1, "grammar_flatness"),
    ("grammar", 5, "grammar_mid_texton"),
    ("grammar", 6, "grammar_high_texture"),
    ("grammar", 7, "grammar_high_suppression"),
]


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def _lowpass(x: torch.Tensor, kernel: int) -> torch.Tensor:
    kernel = max(1, int(kernel))
    if kernel <= 1:
        return x.float()
    if kernel % 2 == 0:
        kernel += 1
    return F.avg_pool2d(x.float(), kernel_size=kernel, stride=1, padding=kernel // 2)


def _band_metrics(delta: torch.Tensor) -> dict[str, float]:
    delta_f = delta.float()
    low = _lowpass(delta_f, 9)
    inner = _lowpass(delta_f, 3)
    mid = inner - low
    high = delta_f - inner
    endpoint_rms = torch.sqrt(delta_f.square().mean())
    high_rms = torch.sqrt(high.square().mean())
    return {
        "endpoint_delta_rms": float(endpoint_rms.item()),
        "endpoint_delta_abs_mean": float(delta_f.abs().mean().item()),
        "low_delta_rms": float(torch.sqrt(low.square().mean()).item()),
        "mid_delta_rms": float(torch.sqrt(mid.square().mean()).item()),
        "high_delta_rms": float(high_rms.item()),
        "high_fraction": float((high_rms / endpoint_rms.clamp_min(1e-8)).item()),
    }


def _debug_scalar(model: torch.nn.Module, key: str) -> float:
    debug = getattr(getattr(model, "blender", None), "last_debug", {}) or {}
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
    device: str,
) -> tuple[torch.nn.Module, dict]:
    ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
    config = copy.deepcopy(ckpt["config"])
    model_cfg = config.setdefault("model", {})
    model_cfg.update(
        {
            "style_tokenizer_enable": True,
            "style_token_identity_dim": int(model_cfg.get("style_token_identity_dim", 16)),
            "style_token_grammar_dim": max(9, int(model_cfg.get("style_token_grammar_dim", 32))),
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


def _apply_stat_vocab(
    model: torch.nn.Module,
    latent_root: Path,
    style_names: list[str],
    *,
    sample_count: int,
    seed: int,
    band_logit_scale: float,
    grammar_scale: float,
    clamp: float,
) -> list[dict[str, Any]]:
    tokenizer = getattr(model, "style_tokenizer", None)
    if tokenizer is None:
        raise RuntimeError("tokenizer was not constructed")
    rng = random.Random(int(seed))
    latent_index = _style_latent_index(latent_root, style_names)
    stats_by_style = [
        _style_stats(_sample_style_tensor(latent_index[style_name], int(sample_count), rng))
        for style_name in style_names
    ]
    recipe = StatProbeRecipe(
        name="operator_binding_stat_vocab",
        band_gain_scale=0.0,
        flatten_strength=0.0,
        flatten_kernel=5,
        band_logit_scale=float(band_logit_scale),
        grammar_scale=float(grammar_scale),
        clamp=float(clamp),
    )
    grammar, band, rows = _build_vocab(
        stats_by_style,
        grammar_dim=int(tokenizer.grammar_vocab.weight.shape[1]),
        band_dim=int(tokenizer.band_vocab.weight.shape[1]),
        recipe=recipe,
    )
    with torch.no_grad():
        tokenizer.grammar_vocab.weight.copy_(
            grammar.to(device=tokenizer.grammar_vocab.weight.device, dtype=tokenizer.grammar_vocab.weight.dtype)
        )
        tokenizer.band_vocab.weight.copy_(
            band.to(device=tokenizer.band_vocab.weight.device, dtype=tokenizer.band_vocab.weight.dtype)
        )
    for row, style_name in zip(rows, style_names):
        row["style_name"] = style_name
    return rows


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
    table = tokenizer.band_vocab.weight if field == "band" else tokenizer.grammar_vocab.weight
    with torch.no_grad():
        table[int(style_id), int(dim)] = value.to(device=table.device, dtype=table.dtype)


def run(args: argparse.Namespace) -> None:
    checkpoint = _resolve_path(args.checkpoint)
    init_style_adapter = _resolve_path(args.init_style_adapter)
    if checkpoint is None or init_style_adapter is None:
        raise ValueError("checkpoint and init-style-adapter are required")
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    latent_root = _resolve_latent_root(ckpt["config"], args.latent_root)
    style_names = [item.strip() for item in args.style_subdirs.split(",") if item.strip()]
    target_style_ids = [int(item.strip()) for item in args.target_style_ids.split(",") if item.strip()]
    rng = random.Random(int(args.seed))

    neutral, config = _load_model(
        checkpoint,
        init_style_adapter=init_style_adapter,
        texture_scale=float(args.texture_scale),
        band_gain_scale=float(args.band_gain_scale),
        flatten_strength=float(args.flatten_strength),
        flatten_kernel=int(args.flatten_kernel),
        device=args.device,
    )
    stat_model, _ = _load_model(
        checkpoint,
        init_style_adapter=init_style_adapter,
        texture_scale=float(args.texture_scale),
        band_gain_scale=float(args.band_gain_scale),
        flatten_strength=float(args.flatten_strength),
        flatten_kernel=int(args.flatten_kernel),
        device=args.device,
    )
    stat_rows = _apply_stat_vocab(
        stat_model,
        latent_root,
        style_names,
        sample_count=int(args.sample_count),
        seed=int(args.seed),
        band_logit_scale=float(args.stat_band_logit_scale),
        grammar_scale=float(args.stat_grammar_scale),
        clamp=float(args.stat_clamp),
    )

    latent_index = _style_latent_index(latent_root, style_names)
    content_pool = [p for style in style_names for p in latent_index[style]]
    rows: list[dict[str, Any]] = []
    perturb_rows: list[dict[str, Any]] = []

    for style_id in target_style_ids:
        style_name = style_names[style_id]
        for batch_idx in range(1, max(1, int(args.num_batches)) + 1):
            content = _sample_latent_batch(content_pool, max(1, int(args.batch_size)), args.device, rng)
            base = _endpoint(neutral, content, style_id, int(args.ode_steps))
            stat = _endpoint(stat_model, content, style_id, int(args.ode_steps))
            row: dict[str, Any] = {
                "kind": "stat_vocab_preview",
                "style_id": style_id,
                "style": style_name,
                "batch": batch_idx,
                "texture_scale": float(args.texture_scale),
                "band_gain_scale": float(args.band_gain_scale),
                "stat_band_logit_scale": float(args.stat_band_logit_scale),
                "stat_grammar_scale": float(args.stat_grammar_scale),
                "grammar_mid_alloc_mean": _debug_scalar(stat_model, "body_transport_adain_grammar_mid_alloc"),
                "grammar_high_alloc_mean": _debug_scalar(stat_model, "body_transport_adain_grammar_high_alloc"),
            }
            row.update(_band_metrics(stat - base))
            rows.append(row)

            for field, dim, label in FIELD_SPECS:
                for sign in (-1.0, 1.0):
                    original = _perturb_token(neutral, field, dim, style_id, sign * float(args.perturb_delta))
                    try:
                        perturbed = _endpoint(neutral, content, style_id, int(args.ode_steps))
                    finally:
                        _restore_token(neutral, field, dim, style_id, original)
                    prow: dict[str, Any] = {
                        "kind": "local_perturbation",
                        "style_id": style_id,
                        "style": style_name,
                        "batch": batch_idx,
                        "field": field,
                        "dim": dim,
                        "label": label,
                        "delta": sign * float(args.perturb_delta),
                        "grammar_mid_alloc_mean": _debug_scalar(neutral, "body_transport_adain_grammar_mid_alloc"),
                        "grammar_high_alloc_mean": _debug_scalar(neutral, "body_transport_adain_grammar_high_alloc"),
                    }
                    prow.update(_band_metrics(perturbed - base))
                    perturb_rows.append(prow)
            del content, base, stat
            if str(args.device).startswith("cuda"):
                torch.cuda.empty_cache()

    all_rows = rows + perturb_rows
    args.out_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(args.out_dir / "operator_binding_rows.csv", all_rows)
    _write_csv(args.out_dir / "stat_vocab_rows.csv", stat_rows)
    manifest = {
        "checkpoint": str(checkpoint),
        "init_style_adapter": str(init_style_adapter),
        "latent_root": str(latent_root),
        "style_names": style_names,
        "target_style_ids": target_style_ids,
        "batch_size": int(args.batch_size),
        "num_batches": int(args.num_batches),
        "ode_steps": int(args.ode_steps),
        "texture_scale": float(args.texture_scale),
        "band_gain_scale": float(args.band_gain_scale),
        "flatten_strength": float(args.flatten_strength),
        "flatten_kernel": int(args.flatten_kernel),
        "stat_band_logit_scale": float(args.stat_band_logit_scale),
        "stat_grammar_scale": float(args.stat_grammar_scale),
        "stat_clamp": float(args.stat_clamp),
        "hypothesis": (
            "Bind grammar[5]/grammar[6] directly to transport-AdaIN mid/high residual gains; "
            "if stat tokens move mid/high endpoint bands, tokenizer has an executable texture operator."
        ),
        "missing_keys_from_source": getattr(neutral, "_tokenizer_load_missing", []),
        "unexpected_keys_from_source": getattr(neutral, "_tokenizer_load_unexpected", []),
        "main_omf_loss_changed": False,
    }
    _write_json(args.out_dir / "manifest.json", manifest)

    top = sorted(all_rows, key=lambda item: float(item.get("endpoint_delta_rms", 0.0)), reverse=True)[:16]
    lines = [
        "# m02 Tokenizer Operator-Binding Diagnostic",
        "",
        "No training. Main OMF loss unchanged.",
        "",
        "One-line hypothesis: grammar[5]/grammar[6] should be executable only if they are hard-bound to the existing m02 mid/high transport-AdaIN residuals.",
        "",
        "## Strongest Responses",
        "",
        "| kind | style | label | delta | endpoint_rms | low | mid | high | high_fraction |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in top:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row.get("kind", "")),
                    str(row.get("style", "")),
                    str(row.get("label", "stat_vocab")),
                    f"{float(row.get('delta', 0.0)):.3f}" if row.get("kind") == "local_perturbation" else "",
                    f"{float(row.get('endpoint_delta_rms', 0.0)):.6f}",
                    f"{float(row.get('low_delta_rms', 0.0)):.6f}",
                    f"{float(row.get('mid_delta_rms', 0.0)):.6f}",
                    f"{float(row.get('high_delta_rms', 0.0)):.6f}",
                    f"{float(row.get('high_fraction', 0.0)):.3f}",
                ]
            )
            + " |"
        )
    lines += [
        "",
        "## Decision Rule",
        "",
        "- If only band_low dominates, the tokenizer is still a color/fog valve.",
        "- If grammar_mid_texton and grammar_high_texture create comparable mid/high motion, the next experiment may train tokenizer fields while freezing m02.",
        "- If stat_vocab_preview is tiny, measured token values are still not aligned to the executable direction.",
    ]
    (args.out_dir / "operator_binding_readout.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(args.out_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--init-style-adapter", type=Path, required=True)
    parser.add_argument("--latent-root", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, default=ROOT / "exp/diagnostics/m02_tokenizer_operator_binding")
    parser.add_argument("--style-subdirs", type=str, default="photo,Hayao,monet,vangogh,cezanne")
    parser.add_argument("--target-style-ids", type=str, default="1,2,3,4")
    parser.add_argument("--sample-count", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=3)
    parser.add_argument("--num-batches", type=int, default=1)
    parser.add_argument("--ode-steps", type=int, default=12)
    parser.add_argument("--band-gain-scale", type=float, default=0.24)
    parser.add_argument("--texture-scale", type=float, default=0.35)
    parser.add_argument("--flatten-strength", type=float, default=0.0)
    parser.add_argument("--flatten-kernel", type=int, default=7)
    parser.add_argument("--stat-band-logit-scale", type=float, default=1.20)
    parser.add_argument("--stat-grammar-scale", type=float, default=1.05)
    parser.add_argument("--stat-clamp", type=float, default=1.65)
    parser.add_argument("--perturb-delta", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=9601)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
