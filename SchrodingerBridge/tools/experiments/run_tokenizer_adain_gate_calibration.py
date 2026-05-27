from __future__ import annotations

"""Tokenizer calibration on the style-normal transport-AdaIN anchor.

Hypothesis: tokenizer should not replace the output head. It should act as a
low-rank valve over the already style-normal transport-AdaIN carrier: band
tokens gate low/mid/high residuals, while grammar tokens optionally suppress
high-frequency drift in flat regions. The backbone and m02 style adapter stay
frozen.
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

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

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


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def _highpass(x: torch.Tensor, kernel: int) -> torch.Tensor:
    k = max(1, int(kernel))
    if k <= 1:
        return x.float()
    if k % 2 == 0:
        k += 1
    low = F.avg_pool2d(x.float(), kernel_size=k, stride=1, padding=k // 2)
    return x.float() - low


def _l2_mean(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return (a.float() - b.float()).square().mean()


@dataclass(frozen=True)
class AdainGateRecipe:
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
    token_l2_weight: float
    highpass_kernel: int
    band_gain_scale: float
    flatten_strength: float
    flatten_kernel: int
    save_every: int = 0


RECIPES = [
    AdainGateRecipe(
        name="ag00_m02_safe_gate",
        iters_per_style=120,
        batch_size=14,
        ode_steps=12,
        lr=1.4e-3,
        swd_weight=0.65,
        hp_swd_weight=0.85,
        anchor_weight=0.22,
        grad_weight=0.14,
        delta_tv_weight=0.05,
        token_l2_weight=0.018,
        highpass_kernel=5,
        band_gain_scale=0.18,
        flatten_strength=0.045,
        flatten_kernel=5,
        save_every=60,
    ),
    AdainGateRecipe(
        name="ag01_m02_style_gate",
        iters_per_style=150,
        batch_size=14,
        ode_steps=12,
        lr=1.2e-3,
        swd_weight=0.78,
        hp_swd_weight=1.05,
        anchor_weight=0.14,
        grad_weight=0.10,
        delta_tv_weight=0.04,
        token_l2_weight=0.014,
        highpass_kernel=5,
        band_gain_scale=0.24,
        flatten_strength=0.070,
        flatten_kernel=7,
        save_every=75,
    ),
]


def _parse_recipes(spec: str) -> list[AdainGateRecipe]:
    if not spec.strip():
        return RECIPES
    keep = {item.strip() for item in spec.split(",") if item.strip()}
    selected = [recipe for recipe in RECIPES if recipe.name in keep]
    if not selected:
        raise ValueError(f"No matching recipes for {spec!r}")
    return selected


def _resolve_path(path: Path | None) -> Path | None:
    if path is None:
        return None
    return path if path.is_absolute() else (ROOT / path).resolve()


def _resolve_latent_root(config: dict, requested: Path | None) -> Path:
    requested = _resolve_path(requested)
    if requested is not None:
        return requested
    data_root = str((config.get("data", {}) or {}).get("data_root", "")).strip()
    if data_root:
        p = Path(data_root)
        return p if p.is_absolute() else (ROOT / p).resolve()
    return ROOT.parent / "latent-256-sd15-ema"


def _read_summary_metrics(summary: dict) -> dict:
    overview = dict(summary.get("analysis", {}).get("all_pairs_overview", {}) or {})
    cross_by_target = dict(summary.get("analysis", {}).get("cross_by_target_style", {}) or {})
    hayao_cross = dict(cross_by_target.get("Hayao", {}) or cross_by_target.get("hayao", {}) or {})
    valid_targets = [
        (str(name), dict(payload))
        for name, payload in cross_by_target.items()
        if isinstance(payload, dict) and payload.get("clip_style") is not None
    ]
    valid_targets.sort(key=lambda item: float(item[1].get("clip_style", float("inf"))))
    weakest = valid_targets[0] if valid_targets else ("", {})
    return {
        "clip_style": overview.get("clip_style", float("nan")),
        "clip_content": overview.get("clip_content", float("nan")),
        "content_lpips": overview.get("content_lpips", float("nan")),
        "ec": overview.get("edge_consistency", overview.get("ec", float("nan"))),
        "hayao_cross_clip_style": hayao_cross.get("clip_style", float("nan")),
        "hayao_cross_content_lpips": hayao_cross.get("content_lpips", float("nan")),
        "weakest_cross_target": weakest[0],
        "weakest_cross_clip_style": weakest[1].get("clip_style", float("nan")),
        "weakest_cross_content_lpips": weakest[1].get("content_lpips", float("nan")),
    }


def _load_tokenizer_adain_model(
    checkpoint: Path,
    *,
    init_style_adapter: Path,
    recipe: AdainGateRecipe,
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
    model._tokenizer_load_missing = list(missing)
    model._tokenizer_load_unexpected = list(unexpected)
    return model, config


def _save_checkpoint(
    path: Path,
    model,
    config: dict,
    *,
    source_checkpoint: Path,
    init_style_adapter: Path,
    recipe: AdainGateRecipe,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": {key: value.detach().cpu() for key, value in model.state_dict().items()},
            "config": config,
            "tokenizer_adain_source_checkpoint": str(source_checkpoint),
            "tokenizer_adain_init_style_adapter": str(init_style_adapter),
            "tokenizer_adain_recipe": recipe.__dict__,
        },
        path,
    )


def run_recipe(
    recipe: AdainGateRecipe,
    *,
    checkpoint: Path,
    init_style_adapter: Path,
    latent_root: Path,
    out_root: Path,
    style_names: list[str],
    target_style_ids: list[int],
    eval_batch_size: int,
    vae_model: str,
    seed: int,
    device: str,
    skip_eval: bool,
    max_iters_per_style: int,
) -> dict:
    rng = random.Random(seed)
    model, config = _load_tokenizer_adain_model(
        checkpoint,
        init_style_adapter=init_style_adapter,
        recipe=recipe,
        device=device,
    )
    teacher, _ = _load_tokenizer_adain_model(
        checkpoint,
        init_style_adapter=init_style_adapter,
        recipe=recipe,
        device=device,
    )
    teacher.eval()
    for param in teacher.parameters():
        param.requires_grad_(False)

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
    optimizer = torch.optim.AdamW(params, lr=recipe.lr, weight_decay=0.0)

    latent_index = _style_latent_index(latent_root, style_names)
    content_pool = [p for style in style_names for p in latent_index[style]]
    transport = SWDTransportCost(config)
    recipe_dir = out_root / recipe.name
    losses: list[dict] = []
    start_time = time.time()
    iters_per_style = min(recipe.iters_per_style, max_iters_per_style) if max_iters_per_style > 0 else recipe.iters_per_style

    for style_id in target_style_ids:
        style_name = style_names[style_id]
        for iteration in range(1, iters_per_style + 1):
            content = _sample_latent_batch(content_pool, recipe.batch_size, device, rng)
            target = _sample_latent_batch(latent_index[style_name], recipe.batch_size, device, rng)
            sid = torch.full((recipe.batch_size,), style_id, dtype=torch.long, device=device)

            optimizer.zero_grad(set_to_none=True)
            pred = _integrate_with_grad(model, content, style_id=sid, num_steps=recipe.ode_steps)
            with torch.no_grad():
                teacher_pred = _integrate_with_grad(teacher, content, style_id=sid, num_steps=recipe.ode_steps)
            swd = transport.aligned_cost(pred, target)
            hp_swd = transport.aligned_cost(_highpass(pred, recipe.highpass_kernel), _highpass(target, recipe.highpass_kernel))
            anchor = _l2_mean(pred, teacher_pred)
            grad = _gradient_cosine_loss(pred, content) if recipe.grad_weight > 0.0 else pred.new_tensor(0.0)
            tv = _tv_loss(pred - content) if recipe.delta_tv_weight > 0.0 else pred.new_tensor(0.0)
            token_l2 = _l2_mean(tokenizer.grammar_vocab.weight, base_grammar) + _l2_mean(tokenizer.band_vocab.weight, base_band)
            loss = (
                recipe.swd_weight * swd
                + recipe.hp_swd_weight * hp_swd
                + recipe.anchor_weight * anchor
                + recipe.grad_weight * grad
                + recipe.delta_tv_weight * tv
                + recipe.token_l2_weight * token_l2
            )
            if not torch.isfinite(loss.detach()):
                raise FloatingPointError(f"Non-finite loss in {recipe.name} style={style_name} iter={iteration}")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            optimizer.step()

            with torch.no_grad():
                gains = 1.0 + torch.tanh(tokenizer.band_vocab.weight[:, :3]) * float(recipe.band_gain_scale)
                grammar_norm = tokenizer.grammar_vocab.weight[style_id].detach().float().norm()
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
                "token_l2": float(token_l2.detach().item()),
                "grammar_norm": float(grammar_norm.item()),
                "gain_low": float(gains[style_id, 0].detach().item()),
                "gain_mid": float(gains[style_id, 1].detach().item()),
                "gain_high": float(gains[style_id, 2].detach().item()),
            }
            losses.append(row)
            if iteration == 1 or iteration % 25 == 0 or iteration == iters_per_style:
                print(
                    f"[{recipe.name}] style={style_name} iter={iteration}/{iters_per_style} "
                    f"loss={row['loss']:.4f} swd={row['swd']:.4f} hp={row['hp_swd']:.4f} "
                    f"anchor={row['anchor']:.5f} gains=({row['gain_low']:.3f},{row['gain_mid']:.3f},{row['gain_high']:.3f})"
                )
            if recipe.save_every > 0 and iteration % recipe.save_every == 0:
                _save_style_adapter(recipe_dir / f"style_adapter_style{style_id}_iter{iteration:04d}.pt", model)
            del content, target, sid, pred, teacher_pred, loss
            if device.startswith("cuda"):
                torch.cuda.empty_cache()

    adapter_path = recipe_dir / "style_adapter.pt"
    checkpoint_path = recipe_dir / "checkpoint_tokenizer_adain_gate.pt"
    _save_style_adapter(adapter_path, model)
    _save_checkpoint(
        checkpoint_path,
        model,
        config,
        source_checkpoint=checkpoint,
        init_style_adapter=init_style_adapter,
        recipe=recipe,
    )
    with (recipe_dir / "calibration_losses.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(losses[0].keys()))
        writer.writeheader()
        writer.writerows(losses)
    _write_json(
        recipe_dir / "calibration_config.json",
        {
            "recipe": recipe.__dict__,
            "checkpoint": str(checkpoint),
            "init_style_adapter": str(init_style_adapter),
            "latent_root": str(latent_root),
            "style_names": style_names,
            "target_style_ids": target_style_ids,
            "effective_iters_per_style": iters_per_style,
            "elapsed_seconds": time.time() - start_time,
            "hypothesis": (
                "Freeze the m02 transport-AdaIN anchor and train only tokenizer grammar/band fields. "
                "The fields gate low/mid/high AdaIN residual bands and apply grammar flattening in flat regions."
            ),
            "missing_keys_from_source": getattr(model, "_tokenizer_load_missing", []),
            "unexpected_keys_from_source": getattr(model, "_tokenizer_load_unexpected", []),
        },
    )

    if skip_eval:
        return {"recipe": recipe.name, "adapter_path": str(adapter_path), "checkpoint": str(checkpoint_path)}

    full_eval_dir = recipe_dir / "full_eval"
    summary = _run_full_eval(
        checkpoint=checkpoint_path,
        style_adapter=adapter_path,
        output_dir=full_eval_dir,
        batch_size=eval_batch_size,
        vae_model=vae_model,
    )
    _write_json(recipe_dir / "full_eval_summary.json", summary)
    return {
        "recipe": recipe.name,
        "adapter_path": str(adapter_path),
        "checkpoint": str(checkpoint_path),
        "full_eval_dir": str(full_eval_dir),
        **_read_summary_metrics(summary),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Tokenizer-gated transport-AdaIN calibration on m02.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--init-style-adapter", type=Path, required=True)
    parser.add_argument("--latent-root", type=Path, default=None)
    parser.add_argument("--out-root", type=Path, default=ROOT / "exp/tokenizer_adain_gate_calibration")
    parser.add_argument("--style-subdirs", type=str, default="photo,Hayao,monet,vangogh,cezanne")
    parser.add_argument("--target-style-ids", type=str, default="1,2,3,4")
    parser.add_argument("--recipes", type=str, default="")
    parser.add_argument("--eval-batch-size", type=int, default=0)
    parser.add_argument("--vae-model", type=str, default="auto")
    parser.add_argument("--seed", type=int, default=42)
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
    rows: list[dict] = []
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
                eval_batch_size=eval_batch_size,
                vae_model=args.vae_model,
                seed=args.seed + idx,
                device=args.device,
                skip_eval=args.skip_eval,
                max_iters_per_style=args.max_iters_per_style,
            )
        )
        with (args.out_root / "tokenizer_adain_gate_results.csv").open("w", newline="", encoding="utf-8") as f:
            fieldnames: list[str] = []
            for row in rows:
                for key in row:
                    if key not in fieldnames:
                        fieldnames.append(key)
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
    _write_json(args.out_root / "run_manifest.json", {"checkpoint": str(checkpoint), "rows": rows})
    print(args.out_root / "tokenizer_adain_gate_results.csv")


if __name__ == "__main__":
    main()
