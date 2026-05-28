from __future__ import annotations

"""Diagnostic-only tiny reader for fixed stat tokenizer fields on m02.

The tokenizer is frozen. The m02 backbone and style adapter are frozen. Only a
zero-initialized token_reader inside transport-AdaIN learns how to interpret
the data-derived tokenizer coordinates.

This script is kept as a recorded negative/control probe. It does not modify
the main OMF loss and should not be used as a scalar-loss route to compensate
for weak tokenizer executability.
"""

import argparse
import copy
import csv
import json
import random
import sys
import time
from dataclasses import dataclass
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
    _memory_tier_eval_batch_size,
    _run_full_eval,
    _sample_latent_batch,
    _save_style_adapter,
    _style_latent_index,
    _tv_loss,
)
from run_style_embedding_mainline_calibration import _apply_style_adapter  # noqa: E402
from run_tokenizer_adain_gate_calibration import (  # noqa: E402
    _highpass,
    _l2_mean,
    _read_summary_metrics,
    _resolve_latent_root,
    _resolve_path,
    _write_json,
)
from run_tokenizer_stat_vocab_probe import (  # noqa: E402
    StatProbeRecipe,
    _build_vocab,
    _sample_style_tensor,
    _style_stats,
)


@dataclass(frozen=True)
class ReaderRecipe:
    name: str
    iters_per_style: int
    batch_size: int
    ode_steps: int
    lr: float
    swd_weight: float
    hp_swd_weight: float
    anchor_weight: float
    grad_weight: float
    delta_tv_weight: float
    highpass_kernel: int
    band_gain_scale: float
    flatten_strength: float
    flatten_kernel: int
    stat_band_logit_scale: float
    stat_grammar_scale: float
    stat_clamp: float
    reader_hidden: int
    reader_scale: float
    save_every: int = 0


RECIPES = [
    ReaderRecipe(
        name="sr00_stat_reader_safe",
        iters_per_style=120,
        batch_size=14,
        ode_steps=12,
        lr=1.2e-3,
        swd_weight=0.74,
        hp_swd_weight=1.00,
        anchor_weight=0.18,
        grad_weight=0.12,
        delta_tv_weight=0.045,
        highpass_kernel=5,
        band_gain_scale=0.22,
        flatten_strength=0.055,
        flatten_kernel=7,
        stat_band_logit_scale=1.00,
        stat_grammar_scale=0.90,
        stat_clamp=1.50,
        reader_hidden=32,
        reader_scale=0.22,
        save_every=60,
    ),
    ReaderRecipe(
        name="sr01_stat_reader_style",
        iters_per_style=150,
        batch_size=14,
        ode_steps=12,
        lr=1.0e-3,
        swd_weight=0.82,
        hp_swd_weight=1.22,
        anchor_weight=0.11,
        grad_weight=0.10,
        delta_tv_weight=0.040,
        highpass_kernel=5,
        band_gain_scale=0.28,
        flatten_strength=0.070,
        flatten_kernel=7,
        stat_band_logit_scale=1.20,
        stat_grammar_scale=1.05,
        stat_clamp=1.65,
        reader_hidden=40,
        reader_scale=0.34,
        save_every=75,
    ),
]


def _parse_recipes(spec: str) -> list[ReaderRecipe]:
    if not spec.strip():
        return RECIPES
    keep = {item.strip() for item in spec.split(",") if item.strip()}
    selected = [recipe for recipe in RECIPES if recipe.name in keep]
    if not selected:
        raise ValueError(f"No matching recipes for {spec!r}")
    return selected


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _load_reader_model(
    checkpoint: Path,
    *,
    init_style_adapter: Path,
    recipe: ReaderRecipe,
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
            "style_token_band_gain_scale": float(recipe.band_gain_scale),
            "style_token_learn_identity": False,
            "style_token_flatten_strength": float(recipe.flatten_strength),
            "style_token_flatten_kernel": int(recipe.flatten_kernel),
            "style_token_adain_gate_enable": True,
            "style_token_reader_enable": True,
            "style_token_reader_hidden": int(recipe.reader_hidden),
            "style_token_reader_scale": float(recipe.reader_scale),
        }
    )
    state = ckpt["model_state_dict"]
    if any(k.startswith("_orig_mod.") for k in state):
        state = {k.replace("_orig_mod.", ""): v for k, v in state.items()}
    model = build_model_from_config(config["model"], use_checkpointing=False).to(device)
    missing, unexpected = model.load_state_dict(state, strict=False)
    unexpected_clean = [key for key in unexpected if not key.startswith("style_tokenizer.") and ".token_reader." not in key]
    if unexpected_clean:
        raise RuntimeError(f"Unexpected non-tokenizer checkpoint keys: {unexpected_clean[:8]}")
    _apply_style_adapter(model, init_style_adapter, device)
    model._tokenizer_load_missing = list(missing)
    model._tokenizer_load_unexpected = list(unexpected)
    return model, config


def _apply_stat_vocab(
    model,
    latent_root: Path,
    style_names: list[str],
    recipe: ReaderRecipe,
    *,
    sample_count: int,
    seed: int,
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
    stat_recipe = StatProbeRecipe(
        name=recipe.name,
        band_gain_scale=recipe.band_gain_scale,
        flatten_strength=recipe.flatten_strength,
        flatten_kernel=recipe.flatten_kernel,
        band_logit_scale=recipe.stat_band_logit_scale,
        grammar_scale=recipe.stat_grammar_scale,
        clamp=recipe.stat_clamp,
    )
    grammar, band, rows = _build_vocab(
        stats_by_style,
        grammar_dim=int(tokenizer.grammar_vocab.weight.shape[1]),
        band_dim=int(tokenizer.band_vocab.weight.shape[1]),
        recipe=stat_recipe,
    )
    with torch.no_grad():
        tokenizer.grammar_vocab.weight.copy_(
            grammar.to(device=tokenizer.grammar_vocab.weight.device, dtype=tokenizer.grammar_vocab.weight.dtype)
        )
        tokenizer.band_vocab.weight.copy_(band.to(device=tokenizer.band_vocab.weight.device, dtype=tokenizer.band_vocab.weight.dtype))
    for row, style_name in zip(rows, style_names):
        row["recipe"] = recipe.name
        row["style_name"] = style_name
    return rows


def _save_checkpoint(
    path: Path,
    model,
    config: dict,
    *,
    source_checkpoint: Path,
    init_style_adapter: Path,
    recipe: ReaderRecipe,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": {key: value.detach().cpu() for key, value in model.state_dict().items()},
            "config": config,
            "tokenizer_stat_reader_source_checkpoint": str(source_checkpoint),
            "tokenizer_stat_reader_init_style_adapter": str(init_style_adapter),
            "tokenizer_stat_reader_recipe": recipe.__dict__,
        },
        path,
    )


def _reader_parameters(model) -> list[torch.nn.Parameter]:
    reader = getattr(getattr(model, "blender", None), "token_reader", None)
    if reader is None:
        raise RuntimeError("model.blender.token_reader was not constructed")
    return list(reader.parameters())


def run_recipe(
    recipe: ReaderRecipe,
    *,
    checkpoint: Path,
    init_style_adapter: Path,
    latent_root: Path,
    out_root: Path,
    style_names: list[str],
    target_style_ids: list[int],
    sample_count: int,
    eval_batch_size: int,
    vae_model: str,
    seed: int,
    device: str,
    skip_eval: bool,
    max_iters_per_style: int,
) -> dict[str, Any]:
    rng = random.Random(int(seed))
    model, config = _load_reader_model(checkpoint, init_style_adapter=init_style_adapter, recipe=recipe, device=device)
    teacher, _ = _load_reader_model(checkpoint, init_style_adapter=init_style_adapter, recipe=recipe, device=device)
    stat_rows = _apply_stat_vocab(model, latent_root, style_names, recipe, sample_count=sample_count, seed=seed)
    _apply_stat_vocab(teacher, latent_root, style_names, recipe, sample_count=sample_count, seed=seed)

    teacher.eval()
    for param in teacher.parameters():
        param.requires_grad_(False)
    model.train()
    for param in model.parameters():
        param.requires_grad_(False)
    params = _reader_parameters(model)
    for param in params:
        param.requires_grad_(True)
    optimizer = torch.optim.AdamW(params, lr=float(recipe.lr), weight_decay=0.0)

    latent_index = _style_latent_index(latent_root, style_names)
    content_pool = [p for style in style_names for p in latent_index[style]]
    transport = SWDTransportCost(config)
    recipe_dir = out_root / recipe.name
    losses: list[dict[str, Any]] = []
    start_time = time.time()
    iters_per_style = min(recipe.iters_per_style, max_iters_per_style) if max_iters_per_style > 0 else recipe.iters_per_style

    for style_id in target_style_ids:
        style_name = style_names[style_id]
        for iteration in range(1, iters_per_style + 1):
            content = _sample_latent_batch(content_pool, int(recipe.batch_size), device, rng)
            target = _sample_latent_batch(latent_index[style_name], int(recipe.batch_size), device, rng)
            sid = torch.full((recipe.batch_size,), int(style_id), dtype=torch.long, device=device)
            optimizer.zero_grad(set_to_none=True)
            pred = _integrate_with_grad(model, content, style_id=sid, num_steps=recipe.ode_steps)
            with torch.no_grad():
                teacher_pred = _integrate_with_grad(teacher, content, style_id=sid, num_steps=recipe.ode_steps)
            swd = transport.aligned_cost(pred, target)
            hp_swd = transport.aligned_cost(_highpass(pred, recipe.highpass_kernel), _highpass(target, recipe.highpass_kernel))
            anchor = _l2_mean(pred, teacher_pred)
            grad = _gradient_cosine_loss(pred, content) if recipe.grad_weight > 0.0 else pred.new_tensor(0.0)
            tv = _tv_loss(pred - content) if recipe.delta_tv_weight > 0.0 else pred.new_tensor(0.0)
            loss = (
                recipe.swd_weight * swd
                + recipe.hp_swd_weight * hp_swd
                + recipe.anchor_weight * anchor
                + recipe.grad_weight * grad
                + recipe.delta_tv_weight * tv
            )
            if not torch.isfinite(loss.detach()):
                raise FloatingPointError(f"Non-finite loss in {recipe.name} style={style_name} iter={iteration}")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            optimizer.step()
            with torch.no_grad():
                reader_norm = torch.sqrt(sum(p.detach().float().square().sum() for p in params))
            row = {
                "recipe": recipe.name,
                "style_id": style_id,
                "style_name": style_name,
                "iter": iteration,
                "loss": float(loss.detach().item()),
                "swd": float(swd.detach().item()),
                "hp_swd": float(hp_swd.detach().item()),
                "anchor": float(anchor.detach().item()),
                "grad": float(grad.detach().item()),
                "tv": float(tv.detach().item()),
                "reader_norm": float(reader_norm.item()),
            }
            losses.append(row)
            if iteration == 1 or iteration % 25 == 0 or iteration == iters_per_style:
                print(
                    f"[{recipe.name}] style={style_name} iter={iteration}/{iters_per_style} "
                    f"loss={row['loss']:.4f} hp={row['hp_swd']:.4f} anchor={row['anchor']:.5f} reader={row['reader_norm']:.4f}"
                )
            if recipe.save_every > 0 and iteration % recipe.save_every == 0:
                _save_style_adapter(recipe_dir / f"style_adapter_style{style_id}_iter{iteration:04d}.pt", model)
            del content, target, sid, pred, teacher_pred, loss
            if device.startswith("cuda"):
                torch.cuda.empty_cache()

    adapter_path = recipe_dir / "style_adapter.pt"
    checkpoint_path = recipe_dir / "checkpoint_tokenizer_stat_reader.pt"
    _save_style_adapter(adapter_path, model)
    _save_checkpoint(
        checkpoint_path,
        model,
        config,
        source_checkpoint=checkpoint,
        init_style_adapter=init_style_adapter,
        recipe=recipe,
    )
    _write_csv(recipe_dir / "stat_vocab_rows.csv", stat_rows)
    _write_csv(recipe_dir / "calibration_losses.csv", losses)
    _write_json(
        recipe_dir / "calibration_config.json",
        {
            "recipe": recipe.__dict__,
            "checkpoint": str(checkpoint),
            "init_style_adapter": str(init_style_adapter),
            "latent_root": str(latent_root),
            "style_names": style_names,
            "target_style_ids": target_style_ids,
            "sample_count": int(sample_count),
            "effective_iters_per_style": iters_per_style,
            "elapsed_seconds": time.time() - start_time,
            "hypothesis": (
                "Freeze the stat tokenizer and the m02 carrier; train only a zero-initialized token_reader "
                "so the carrier learns to interpret measured style coordinates."
            ),
            "missing_keys_from_source": getattr(model, "_tokenizer_load_missing", []),
            "unexpected_keys_from_source": getattr(model, "_tokenizer_load_unexpected", []),
        },
    )

    row: dict[str, Any] = {"recipe": recipe.name, "adapter_path": str(adapter_path), "checkpoint": str(checkpoint_path)}
    if skip_eval:
        return row
    full_eval_dir = recipe_dir / "full_eval"
    summary = _run_full_eval(
        checkpoint=checkpoint_path,
        style_adapter=adapter_path,
        output_dir=full_eval_dir,
        batch_size=eval_batch_size,
        vae_model=vae_model,
    )
    _write_json(recipe_dir / "full_eval_summary.json", summary)
    row.update({"full_eval_dir": str(full_eval_dir), **_read_summary_metrics(summary)})
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--init-style-adapter", type=Path, required=True)
    parser.add_argument("--latent-root", type=Path, default=None)
    parser.add_argument("--out-root", type=Path, default=ROOT / "exp/tokenizer_stat_reader_probe")
    parser.add_argument("--style-subdirs", type=str, default="photo,Hayao,monet,vangogh,cezanne")
    parser.add_argument("--target-style-ids", type=str, default="1,2,3,4")
    parser.add_argument("--recipes", type=str, default="")
    parser.add_argument("--sample-count", type=int, default=256)
    parser.add_argument("--eval-batch-size", type=int, default=0)
    parser.add_argument("--vae-model", type=str, default="auto")
    parser.add_argument("--seed", type=int, default=9503)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--skip-eval", action="store_true")
    parser.add_argument("--max-iters-per-style", type=int, default=0)
    args = parser.parse_args()

    checkpoint = _resolve_path(args.checkpoint)
    init_style_adapter = _resolve_path(args.init_style_adapter)
    if checkpoint is None or init_style_adapter is None:
        raise ValueError("checkpoint and init-style-adapter are required")
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    latent_root = _resolve_latent_root(ckpt["config"], args.latent_root)
    style_names = [item.strip() for item in args.style_subdirs.split(",") if item.strip()]
    target_style_ids = [int(item.strip()) for item in args.target_style_ids.split(",") if item.strip()]
    recipes = _parse_recipes(args.recipes)
    eval_batch_size = _memory_tier_eval_batch_size(args.device, args.eval_batch_size if args.eval_batch_size > 0 else None)

    args.out_root.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for idx, recipe in enumerate(recipes):
        rows.append(
            run_recipe(
                recipe,
                checkpoint=checkpoint,
                init_style_adapter=init_style_adapter,
                latent_root=latent_root,
                out_root=args.out_root,
                style_names=style_names,
                target_style_ids=target_style_ids,
                sample_count=int(args.sample_count),
                eval_batch_size=eval_batch_size,
                vae_model=args.vae_model,
                seed=int(args.seed) + idx,
                device=args.device,
                skip_eval=bool(args.skip_eval),
                max_iters_per_style=int(args.max_iters_per_style),
            )
        )
        _write_csv(args.out_root / "tokenizer_stat_reader_results.csv", rows)
    _write_json(args.out_root / "run_manifest.json", {"checkpoint": str(checkpoint), "rows": rows})
    print(args.out_root / "tokenizer_stat_reader_results.csv")


if __name__ == "__main__":
    main()
