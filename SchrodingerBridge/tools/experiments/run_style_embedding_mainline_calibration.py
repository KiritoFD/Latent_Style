from __future__ import annotations

import argparse
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

from ot_cost import SWDTransportCost  # noqa: E402
from run_style_embedding_distill import (  # noqa: E402
    _gradient_cosine_loss,
    _integrate_with_grad,
    _load_checkpoint_model,
    _memory_tier_eval_batch_size,
    _run_full_eval,
    _sample_latent_batch,
    _save_style_adapter,
    _style_latent_index,
    _tv_loss,
)


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


@dataclass(frozen=True)
class MainlineStyleRecipe:
    name: str
    optimize_spatial: bool
    iters_per_style: int
    ode_steps: int
    batch_size: int
    lr: float
    swd_weight: float
    anchor_weight: float
    grad_weight: float
    delta_tv_weight: float
    emb_l2_weight: float
    spatial_l2_weight: float
    highpass_kernel: int
    save_every: int = 0
    optimize_style_emb: bool = True
    optimize_tokenizer: bool = False
    token_l2_weight: float = 0.0


RECIPES = [
    MainlineStyleRecipe(
        name="m00_emb_swd_anchor",
        optimize_spatial=False,
        iters_per_style=140,
        ode_steps=12,
        batch_size=16,
        lr=3e-3,
        swd_weight=1.0,
        anchor_weight=0.25,
        grad_weight=0.15,
        delta_tv_weight=0.05,
        emb_l2_weight=0.05,
        spatial_l2_weight=0.0,
        highpass_kernel=1,
        save_every=70,
    ),
    MainlineStyleRecipe(
        name="m01_embspatial_swd_anchor",
        optimize_spatial=True,
        iters_per_style=160,
        ode_steps=12,
        batch_size=14,
        lr=2e-3,
        swd_weight=1.0,
        anchor_weight=0.35,
        grad_weight=0.25,
        delta_tv_weight=0.10,
        emb_l2_weight=0.08,
        spatial_l2_weight=0.02,
        highpass_kernel=1,
        save_every=80,
    ),
    MainlineStyleRecipe(
        name="m02_embspatial_highpass_style",
        optimize_spatial=True,
        iters_per_style=180,
        ode_steps=12,
        batch_size=14,
        lr=2e-3,
        swd_weight=1.2,
        anchor_weight=0.18,
        grad_weight=0.20,
        delta_tv_weight=0.08,
        emb_l2_weight=0.06,
        spatial_l2_weight=0.02,
        highpass_kernel=5,
        save_every=90,
    ),
    MainlineStyleRecipe(
        name="m03_m02_styleboost_balanced",
        optimize_spatial=True,
        iters_per_style=120,
        ode_steps=12,
        batch_size=14,
        lr=1.0e-3,
        swd_weight=1.65,
        anchor_weight=0.12,
        grad_weight=0.12,
        delta_tv_weight=0.05,
        emb_l2_weight=0.025,
        spatial_l2_weight=0.010,
        highpass_kernel=5,
        save_every=60,
    ),
    MainlineStyleRecipe(
        name="m04_m02_styleboost_loose",
        optimize_spatial=True,
        iters_per_style=150,
        ode_steps=12,
        batch_size=14,
        lr=9.0e-4,
        swd_weight=2.20,
        anchor_weight=0.055,
        grad_weight=0.08,
        delta_tv_weight=0.035,
        emb_l2_weight=0.018,
        spatial_l2_weight=0.008,
        highpass_kernel=5,
        save_every=75,
    ),
    MainlineStyleRecipe(
        name="m05_m02_midcolor_push",
        optimize_spatial=True,
        iters_per_style=130,
        ode_steps=12,
        batch_size=14,
        lr=8.0e-4,
        swd_weight=1.85,
        anchor_weight=0.08,
        grad_weight=0.10,
        delta_tv_weight=0.04,
        emb_l2_weight=0.020,
        spatial_l2_weight=0.010,
        highpass_kernel=3,
        save_every=65,
    ),
    MainlineStyleRecipe(
        name="m10_token_vocab_swd_anchor",
        optimize_spatial=False,
        iters_per_style=120,
        ode_steps=12,
        batch_size=16,
        lr=2.0e-3,
        swd_weight=1.10,
        anchor_weight=0.22,
        grad_weight=0.14,
        delta_tv_weight=0.05,
        emb_l2_weight=0.0,
        spatial_l2_weight=0.0,
        highpass_kernel=3,
        save_every=60,
        optimize_style_emb=False,
        optimize_tokenizer=True,
        token_l2_weight=0.030,
    ),
    MainlineStyleRecipe(
        name="m11_token_vocab_stylepush",
        optimize_spatial=False,
        iters_per_style=140,
        ode_steps=12,
        batch_size=16,
        lr=2.5e-3,
        swd_weight=1.55,
        anchor_weight=0.12,
        grad_weight=0.10,
        delta_tv_weight=0.04,
        emb_l2_weight=0.0,
        spatial_l2_weight=0.0,
        highpass_kernel=5,
        save_every=70,
        optimize_style_emb=False,
        optimize_tokenizer=True,
        token_l2_weight=0.020,
    ),
]


def _parse_recipes(spec: str) -> list[MainlineStyleRecipe]:
    if not spec.strip():
        return RECIPES
    keep = {item.strip() for item in spec.split(",") if item.strip()}
    selected = [recipe for recipe in RECIPES if recipe.name in keep]
    if not selected:
        raise ValueError(f"No matching recipes for {spec!r}")
    return selected


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
    weakest_target = valid_targets[0] if valid_targets else ("", {})
    return {
        "clip_dir": overview.get("clip_dir", float("nan")),
        "clip_style": overview.get("clip_style", float("nan")),
        "clip_content": overview.get("clip_content", float("nan")),
        "content_lpips": overview.get("content_lpips", float("nan")),
        "ec": overview.get("edge_consistency", overview.get("ec", float("nan"))),
        "classifier_acc": overview.get("classifier_acc", float("nan")),
        "hayao_cross_clip_style": hayao_cross.get("clip_style", float("nan")),
        "hayao_cross_content_lpips": hayao_cross.get("content_lpips", float("nan")),
        "weakest_cross_target": weakest_target[0],
        "weakest_cross_clip_style": weakest_target[1].get("clip_style", float("nan")),
        "weakest_cross_content_lpips": weakest_target[1].get("content_lpips", float("nan")),
    }


def _resolve_latent_root(checkpoint_config: dict, requested: Path | None) -> Path:
    if requested is not None:
        return requested
    data_root = str((checkpoint_config.get("data", {}) or {}).get("data_root", "")).strip()
    if data_root:
        p = Path(data_root)
        if not p.is_absolute():
            p = (ROOT / p).resolve()
        return p
    return ROOT.parent / "latent-256"


def _l2_mean(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return (a.float() - b.float()).square().mean()


def _apply_style_adapter(model, adapter_path: Path, device: str) -> None:
    adapter = torch.load(adapter_path, map_location=device, weights_only=False)
    with torch.no_grad():
        style_emb = adapter.get("style_emb.weight")
        if style_emb is None:
            style_emb = adapter.get("style_emb.mu")
        if style_emb is not None:
            model.style_emb.weight.copy_(
                style_emb.to(device=model.style_emb.weight.device, dtype=model.style_emb.weight.dtype)
            )
        style_spatial = adapter.get("style_spatial_id_16")
        if style_spatial is not None and hasattr(model, "style_spatial_id_16"):
            model.style_spatial_id_16.copy_(
                style_spatial.to(device=model.style_spatial_id_16.device, dtype=model.style_spatial_id_16.dtype)
            )
        tokenizer = getattr(model, "style_tokenizer", None)
        if tokenizer is not None:
            grammar = adapter.get("style_tokenizer.grammar_vocab.weight")
            if grammar is not None:
                tokenizer.grammar_vocab.weight.copy_(
                    grammar.to(device=tokenizer.grammar_vocab.weight.device, dtype=tokenizer.grammar_vocab.weight.dtype)
                )
            band = adapter.get("style_tokenizer.band_vocab.weight")
            if band is not None:
                tokenizer.band_vocab.weight.copy_(
                    band.to(device=tokenizer.band_vocab.weight.device, dtype=tokenizer.band_vocab.weight.dtype)
                )
            identity = adapter.get("style_tokenizer.identity_vocab")
            if identity is not None:
                target = getattr(tokenizer, "identity_vocab", None)
                if torch.is_tensor(target) and target.shape == identity.shape:
                    target.copy_(identity.to(device=target.device, dtype=target.dtype))


def run_recipe(
    recipe: MainlineStyleRecipe,
    *,
    checkpoint: Path,
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
    init_style_adapter: Path | None,
) -> dict:
    rng = random.Random(seed)
    model, config = _load_checkpoint_model(checkpoint, device)
    teacher, _ = _load_checkpoint_model(checkpoint, device)
    if init_style_adapter is not None:
        _apply_style_adapter(model, init_style_adapter, device)
        _apply_style_adapter(teacher, init_style_adapter, device)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad_(False)

    model.train()
    for p in model.parameters():
        p.requires_grad_(False)
    model.style_emb.weight.requires_grad_(recipe.optimize_style_emb)
    if hasattr(model, "style_spatial_id_16"):
        model.style_spatial_id_16.requires_grad_(recipe.optimize_spatial)
    tokenizer = getattr(model, "style_tokenizer", None)
    if recipe.optimize_tokenizer:
        if tokenizer is None:
            raise RuntimeError(f"{recipe.name} requires style_tokenizer_enable=True in the checkpoint config")
        tokenizer.grammar_vocab.weight.requires_grad_(True)
        tokenizer.band_vocab.weight.requires_grad_(True)

    base_style_emb = model.style_emb.weight.detach().clone()
    base_style_spatial = (
        model.style_spatial_id_16.detach().clone()
        if recipe.optimize_spatial and hasattr(model, "style_spatial_id_16")
        else None
    )
    base_grammar = (
        tokenizer.grammar_vocab.weight.detach().clone()
        if recipe.optimize_tokenizer and tokenizer is not None
        else None
    )
    base_band = (
        tokenizer.band_vocab.weight.detach().clone()
        if recipe.optimize_tokenizer and tokenizer is not None
        else None
    )

    params = [model.style_emb.weight] if recipe.optimize_style_emb else []
    if recipe.optimize_spatial and hasattr(model, "style_spatial_id_16"):
        params.append(model.style_spatial_id_16)
    if recipe.optimize_tokenizer and tokenizer is not None:
        params.extend([tokenizer.grammar_vocab.weight, tokenizer.band_vocab.weight])
    if not params:
        raise RuntimeError(f"{recipe.name} selected no trainable style parameters")
    optimizer = torch.optim.AdamW(params, lr=recipe.lr, weight_decay=0.0)

    latent_index = _style_latent_index(latent_root, style_names)
    content_pool = [p for style in style_names for p in latent_index[style]]
    transport = SWDTransportCost(config)

    losses: list[dict] = []
    recipe_dir = out_root / recipe.name
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

            pred_for_swd = _highpass(pred, recipe.highpass_kernel)
            target_for_swd = _highpass(target, recipe.highpass_kernel)
            swd = transport.aligned_cost(pred_for_swd, target_for_swd)
            anchor = _l2_mean(pred, teacher_pred)
            grad = _gradient_cosine_loss(pred, content) if recipe.grad_weight > 0.0 else pred.new_tensor(0.0)
            tv = _tv_loss(pred - content) if recipe.delta_tv_weight > 0.0 else pred.new_tensor(0.0)
            emb_l2 = _l2_mean(model.style_emb.weight, base_style_emb)
            spatial_l2 = (
                _l2_mean(model.style_spatial_id_16, base_style_spatial)
                if base_style_spatial is not None and recipe.spatial_l2_weight > 0.0
                else pred.new_tensor(0.0)
            )
            token_l2 = pred.new_tensor(0.0)
            if recipe.token_l2_weight > 0.0 and tokenizer is not None and base_grammar is not None and base_band is not None:
                token_l2 = _l2_mean(tokenizer.grammar_vocab.weight, base_grammar) + _l2_mean(tokenizer.band_vocab.weight, base_band)

            loss = (
                recipe.swd_weight * swd
                + recipe.anchor_weight * anchor
                + recipe.grad_weight * grad
                + recipe.delta_tv_weight * tv
                + recipe.emb_l2_weight * emb_l2
                + recipe.spatial_l2_weight * spatial_l2
                + recipe.token_l2_weight * token_l2
            )
            if not torch.isfinite(loss.detach()):
                raise FloatingPointError(f"Non-finite loss in {recipe.name} style={style_name} iter={iteration}")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            optimizer.step()

            row = {
                "recipe": recipe.name,
                "style_id": style_id,
                "style_name": style_name,
                "iter": iteration,
                "loss": float(loss.detach().item()),
                "swd": float(swd.detach().item()),
                "anchor": float(anchor.detach().item()),
                "grad": float(grad.detach().item()),
                "tv": float(tv.detach().item()),
                "emb_l2": float(emb_l2.detach().item()),
                "spatial_l2": float(spatial_l2.detach().item()),
                "token_l2": float(token_l2.detach().item()),
            }
            losses.append(row)

            if iteration == 1 or iteration % 25 == 0 or iteration == iters_per_style:
                print(
                    f"[{recipe.name}] style={style_name} iter={iteration}/{iters_per_style} "
                    f"loss={row['loss']:.4f} swd={row['swd']:.4f} anchor={row['anchor']:.5f} "
                    f"grad={row['grad']:.4f}"
                )
            if recipe.save_every > 0 and iteration % recipe.save_every == 0:
                _save_style_adapter(recipe_dir / f"style_adapter_style{style_id}_iter{iteration:04d}.pt", model)

            del content, target, sid, pred, teacher_pred, loss
            if device.startswith("cuda"):
                torch.cuda.empty_cache()

    adapter_path = recipe_dir / "style_adapter.pt"
    _save_style_adapter(adapter_path, model)
    with (recipe_dir / "calibration_losses.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(losses[0].keys()))
        writer.writeheader()
        writer.writerows(losses)
    _write_json(
        recipe_dir / "calibration_config.json",
        {
            "recipe": recipe.__dict__,
            "checkpoint": str(checkpoint),
            "init_style_adapter": str(init_style_adapter) if init_style_adapter is not None else "",
            "latent_root": str(latent_root),
            "style_names": style_names,
            "target_style_ids": target_style_ids,
            "effective_iters_per_style": iters_per_style,
            "elapsed_seconds": time.time() - start_time,
            "hypothesis": (
                "Full-train style embedding calibration can raise visible style only if style conditioning "
                "contains unused carrier capacity; teacher endpoint anchoring tests that without external supervision."
            ),
        },
    )

    if skip_eval:
        return {
            "recipe": recipe.name,
            "adapter_path": str(adapter_path),
            "full_eval_dir": "",
            "clip_dir": float("nan"),
            "clip_style": float("nan"),
            "clip_content": float("nan"),
            "content_lpips": float("nan"),
            "ec": float("nan"),
            "classifier_acc": float("nan"),
        }

    full_eval_dir = recipe_dir / "full_eval"
    summary = _run_full_eval(
        checkpoint=checkpoint,
        style_adapter=adapter_path,
        output_dir=full_eval_dir,
        batch_size=eval_batch_size,
        vae_model=vae_model,
    )
    _write_json(recipe_dir / "full_eval_summary.json", summary)
    metrics = _read_summary_metrics(summary)
    return {
        "recipe": recipe.name,
        "adapter_path": str(adapter_path),
        "full_eval_dir": str(full_eval_dir),
        **metrics,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Full-train style_emb calibration with mainline checkpoint anchoring.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--latent-root", type=Path, default=None)
    parser.add_argument("--out-root", type=Path, default=ROOT / "exp/style_embedding_mainline_calibration")
    parser.add_argument("--style-subdirs", type=str, default="photo,Hayao,monet,vangogh,cezanne")
    parser.add_argument("--target-style-ids", type=str, default="1,2,3,4")
    parser.add_argument("--recipes", type=str, default="")
    parser.add_argument("--eval-batch-size", type=int, default=0)
    parser.add_argument("--vae-model", type=str, default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--skip-eval", action="store_true", help="Only fit and save adapters; skip 750-image eval.")
    parser.add_argument("--max-iters-per-style", type=int, default=0, help="Short-run override for smoke tests.")
    parser.add_argument("--init-style-adapter", type=Path, default=None, help="Optional adapter used as the initialization and teacher anchor.")
    args = parser.parse_args()

    _, config = _load_checkpoint_model(args.checkpoint, args.device)
    latent_root = _resolve_latent_root(config, args.latent_root)
    style_names = [item.strip() for item in args.style_subdirs.split(",") if item.strip()]
    target_style_ids = [int(item.strip()) for item in args.target_style_ids.split(",") if item.strip()]
    recipes = _parse_recipes(args.recipes)
    eval_batch_size = _memory_tier_eval_batch_size(
        args.device,
        args.eval_batch_size if args.eval_batch_size > 0 else None,
    )
    args.out_root.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    for recipe in recipes:
        row = run_recipe(
            recipe,
            checkpoint=args.checkpoint,
            latent_root=latent_root,
            out_root=args.out_root,
            style_names=style_names,
            target_style_ids=target_style_ids,
            eval_batch_size=eval_batch_size,
            vae_model=args.vae_model,
            seed=args.seed,
            device=args.device,
            skip_eval=bool(args.skip_eval),
            max_iters_per_style=int(args.max_iters_per_style),
            init_style_adapter=args.init_style_adapter,
        )
        rows.append(row)

    summary_csv = args.out_root / "mainline_style_emb_results.csv"
    with summary_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    _write_json(args.out_root / "summary.json", {"rows": rows})
    print(f"Saved mainline style_emb calibration summary to {summary_csv}")


if __name__ == "__main__":
    main()
