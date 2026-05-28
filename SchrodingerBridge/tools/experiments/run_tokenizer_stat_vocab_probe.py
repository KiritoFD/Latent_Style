from __future__ import annotations

"""Stat-initialized tokenizer probe on the m02 transport-AdaIN anchor.

This is not a loss edit and not a training run. It turns measurable style-pool
statistics into tokenizer fields, then evaluates whether the proven m02 carrier
can read those fields without falling into the hazy factorized failure mode.
"""

import argparse
import csv
import json
import math
import random
import sys
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

from run_style_embedding_distill import (  # noqa: E402
    _load_latent,
    _memory_tier_eval_batch_size,
    _run_full_eval,
    _save_style_adapter,
    _style_latent_index,
)
from run_tokenizer_adain_gate_calibration import (  # noqa: E402
    AdainGateRecipe,
    _load_tokenizer_adain_model,
    _read_summary_metrics,
    _resolve_latent_root,
    _resolve_path,
    _save_checkpoint,
    _write_json,
)


@dataclass(frozen=True)
class StatProbeRecipe:
    name: str
    band_gain_scale: float
    flatten_strength: float
    flatten_kernel: int
    band_logit_scale: float
    grammar_scale: float
    clamp: float


RECIPES = [
    StatProbeRecipe(
        name="sv00_stat_m02_conservative",
        band_gain_scale=0.18,
        flatten_strength=0.045,
        flatten_kernel=5,
        band_logit_scale=0.75,
        grammar_scale=0.70,
        clamp=1.25,
    ),
    StatProbeRecipe(
        name="sv01_stat_m02_balanced",
        band_gain_scale=0.24,
        flatten_strength=0.070,
        flatten_kernel=7,
        band_logit_scale=1.10,
        grammar_scale=1.00,
        clamp=1.60,
    ),
]


def _parse_recipes(spec: str) -> list[StatProbeRecipe]:
    if not spec.strip():
        return RECIPES
    keep = {item.strip() for item in spec.split(",") if item.strip()}
    selected = [recipe for recipe in RECIPES if recipe.name in keep]
    if not selected:
        raise ValueError(f"No matching recipes for {spec!r}")
    return selected


def _as_adain_recipe(recipe: StatProbeRecipe) -> AdainGateRecipe:
    return AdainGateRecipe(
        name=recipe.name,
        iters_per_style=0,
        batch_size=1,
        ode_steps=12,
        lr=0.0,
        swd_weight=0.0,
        hp_swd_weight=0.0,
        anchor_weight=0.0,
        grad_weight=0.0,
        delta_tv_weight=0.0,
        token_l2_weight=0.0,
        highpass_kernel=5,
        band_gain_scale=recipe.band_gain_scale,
        flatten_strength=recipe.flatten_strength,
        flatten_kernel=recipe.flatten_kernel,
        save_every=0,
    )


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


def _lowpass(x: torch.Tensor, kernel: int) -> torch.Tensor:
    kernel = max(1, int(kernel))
    if kernel <= 1:
        return x.float()
    if kernel % 2 == 0:
        kernel += 1
    return F.avg_pool2d(x.float(), kernel_size=kernel, stride=1, padding=kernel // 2)


def _sobel_energy(x: torch.Tensor) -> torch.Tensor:
    channels = int(x.shape[1])
    kx = x.new_tensor([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]]).view(1, 1, 3, 3)
    ky = x.new_tensor([[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]]).view(1, 1, 3, 3)
    kx = kx.expand(channels, 1, 3, 3).contiguous()
    ky = ky.expand(channels, 1, 3, 3).contiguous()
    gx = F.conv2d(x.float(), kx, padding=1, groups=channels)
    gy = F.conv2d(x.float(), ky, padding=1, groups=channels)
    return torch.sqrt(gx.square() + gy.square() + 1e-12)


def _sample_style_tensor(paths: list[Path], sample_count: int, rng: random.Random) -> torch.Tensor:
    count = max(1, min(int(sample_count), len(paths)))
    chosen = rng.sample(paths, count) if count < len(paths) else list(paths)
    return torch.cat([_load_latent(path) for path in chosen], dim=0).float()


def _style_stats(latents: torch.Tensor) -> dict[str, torch.Tensor]:
    low = _lowpass(latents, 9)
    inner = _lowpass(latents, 3)
    mid = inner - low
    high = latents.float() - inner
    bands = torch.stack(
        [
            low.var(unbiased=False),
            mid.var(unbiased=False),
            high.var(unbiased=False),
        ]
    )
    band_ratio = bands / bands.sum().clamp_min(1e-12)
    grad = _sobel_energy(latents)
    flatness = 1.0 / (1.0 + high.abs().mean().clamp_min(1e-8))
    contour_focus = grad.flatten(1).quantile(0.90, dim=1).mean() / grad.mean().clamp_min(1e-8)
    return {
        "band_ratio": band_ratio,
        "high_abs": high.abs().mean(),
        "mid_abs": mid.abs().mean(),
        "flatness": flatness,
        "contour_focus": contour_focus,
    }


def _zscore(values: torch.Tensor) -> torch.Tensor:
    return (values - values.mean()) / values.std(unbiased=False).clamp_min(1e-6)


def _build_vocab(
    stats_by_style: list[dict[str, torch.Tensor]],
    *,
    grammar_dim: int,
    band_dim: int,
    recipe: StatProbeRecipe,
) -> tuple[torch.Tensor, torch.Tensor, list[dict[str, Any]]]:
    ratios = torch.stack([item["band_ratio"] for item in stats_by_style], dim=0).float()
    high_abs = torch.stack([item["high_abs"] for item in stats_by_style]).float()
    mid_abs = torch.stack([item["mid_abs"] for item in stats_by_style]).float()
    flatness = torch.stack([item["flatness"] for item in stats_by_style]).float()
    contour = torch.stack([item["contour_focus"] for item in stats_by_style]).float()

    ref_ratio = ratios[1:].mean(dim=0).clamp_min(1e-8)
    band_logits = torch.log(ratios.clamp_min(1e-8) / ref_ratio.view(1, 3)) * float(recipe.band_logit_scale)
    if band_dim > 3:
        band_logits = F.pad(band_logits, (0, band_dim - 3))
    elif band_dim < 3:
        band_logits = band_logits[:, :band_dim]
    band_logits = band_logits.clamp(-float(recipe.clamp), float(recipe.clamp))

    grammar = torch.zeros(len(stats_by_style), int(grammar_dim), dtype=torch.float32)
    gscale = float(recipe.grammar_scale)
    high_z = _zscore(high_abs)
    mid_z = _zscore(mid_abs)
    flat_z = _zscore(flatness)
    contour_z = _zscore(contour)
    high_ref = high_abs[1:].mean().clamp_min(1e-8)
    flat_need = torch.log(high_ref / high_abs.clamp_min(1e-8))
    if grammar_dim > 1:
        grammar[:, 1] = (flat_z + flat_need) * 0.5 * gscale
    if grammar_dim > 2:
        grammar[:, 2] = contour_z * gscale
    if grammar_dim > 5:
        grammar[:, 5] = mid_z * gscale
    if grammar_dim > 6:
        grammar[:, 6] = high_z * gscale
    if grammar_dim > 7:
        grammar[:, 7] = flat_need * gscale
    grammar = grammar.clamp(-float(recipe.clamp), float(recipe.clamp))
    grammar[0].zero_()
    band_logits[0].zero_()

    rows: list[dict[str, Any]] = []
    for idx, stat in enumerate(stats_by_style):
        row: dict[str, Any] = {
            "style_id": idx,
            "band_low_ratio": float(stat["band_ratio"][0]),
            "band_mid_ratio": float(stat["band_ratio"][1]),
            "band_high_ratio": float(stat["band_ratio"][2]),
            "high_abs": float(stat["high_abs"]),
            "mid_abs": float(stat["mid_abs"]),
            "flatness": float(stat["flatness"]),
            "contour_focus": float(stat["contour_focus"]),
        }
        for j in range(min(3, band_logits.shape[1])):
            row[f"band_logit_{j}"] = float(band_logits[idx, j])
        for j in range(min(9, grammar.shape[1])):
            row[f"grammar_{j}"] = float(grammar[idx, j])
        rows.append(row)
    return grammar, band_logits, rows


def _read_summary(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def run_recipe(
    recipe: StatProbeRecipe,
    *,
    checkpoint: Path,
    init_style_adapter: Path,
    latent_root: Path,
    out_root: Path,
    style_names: list[str],
    sample_count: int,
    eval_batch_size: int,
    vae_model: str,
    seed: int,
    device: str,
    skip_eval: bool,
) -> dict[str, Any]:
    adain_recipe = _as_adain_recipe(recipe)
    model, config = _load_tokenizer_adain_model(
        checkpoint,
        init_style_adapter=init_style_adapter,
        recipe=adain_recipe,
        device=device,
    )
    tokenizer = getattr(model, "style_tokenizer", None)
    if tokenizer is None:
        raise RuntimeError("tokenizer was not constructed")

    rng = random.Random(int(seed))
    latent_index = _style_latent_index(latent_root, style_names)
    stats_by_style = [
        _style_stats(_sample_style_tensor(latent_index[style_name], int(sample_count), rng))
        for style_name in style_names
    ]
    grammar, band_logits, stat_rows = _build_vocab(
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
            band_logits.to(device=tokenizer.band_vocab.weight.device, dtype=tokenizer.band_vocab.weight.dtype)
        )

    recipe_dir = out_root / recipe.name
    adapter_path = recipe_dir / "style_adapter.pt"
    checkpoint_path = recipe_dir / "checkpoint_tokenizer_stat_vocab.pt"
    _save_style_adapter(adapter_path, model)
    _save_checkpoint(
        checkpoint_path,
        model,
        config,
        source_checkpoint=checkpoint,
        init_style_adapter=init_style_adapter,
        recipe=adain_recipe,
    )
    for row, style_name in zip(stat_rows, style_names):
        row["recipe"] = recipe.name
        row["style_name"] = style_name
    _write_csv(recipe_dir / "stat_vocab_rows.csv", stat_rows)
    _write_json(
        recipe_dir / "stat_vocab_config.json",
        {
            "recipe": recipe.__dict__,
            "checkpoint": str(checkpoint),
            "init_style_adapter": str(init_style_adapter),
            "latent_root": str(latent_root),
            "sample_count": int(sample_count),
            "style_names": style_names,
            "hypothesis": (
                "A style tokenizer should first encode measurable training-pool frequency coordinates. "
                "This probe initializes those coordinates without optimization and tests whether m02 can read them."
            ),
            "missing_keys_from_source": getattr(model, "_tokenizer_load_missing", []),
            "unexpected_keys_from_source": getattr(model, "_tokenizer_load_unexpected", []),
        },
    )

    row: dict[str, Any] = {
        "recipe": recipe.name,
        "adapter_path": str(adapter_path),
        "checkpoint": str(checkpoint_path),
    }
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
    parser.add_argument("--out-root", type=Path, default=ROOT / "exp/tokenizer_stat_vocab_probe")
    parser.add_argument("--style-subdirs", type=str, default="photo,Hayao,monet,vangogh,cezanne")
    parser.add_argument("--recipes", type=str, default="")
    parser.add_argument("--sample-count", type=int, default=256)
    parser.add_argument("--eval-batch-size", type=int, default=0)
    parser.add_argument("--vae-model", type=str, default="auto")
    parser.add_argument("--seed", type=int, default=9401)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--skip-eval", action="store_true")
    args = parser.parse_args()

    checkpoint = _resolve_path(args.checkpoint)
    init_style_adapter = _resolve_path(args.init_style_adapter)
    if checkpoint is None or init_style_adapter is None:
        raise ValueError("checkpoint and init-style-adapter are required")
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    latent_root = _resolve_latent_root(ckpt["config"], args.latent_root)
    style_names = [item.strip() for item in args.style_subdirs.split(",") if item.strip()]
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
                sample_count=int(args.sample_count),
                eval_batch_size=eval_batch_size,
                vae_model=args.vae_model,
                seed=int(args.seed) + idx,
                device=args.device,
                skip_eval=bool(args.skip_eval),
            )
        )
        _write_csv(args.out_root / "tokenizer_stat_vocab_results.csv", rows)
    _write_json(args.out_root / "run_manifest.json", {"checkpoint": str(checkpoint), "rows": rows})
    print(args.out_root / "tokenizer_stat_vocab_results.csv")


if __name__ == "__main__":
    main()
