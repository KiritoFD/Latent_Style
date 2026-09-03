from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from ot_cost import SWDTransportCost  # noqa: E402
from run_style_embedding_distill import (  # noqa: E402
    _gradient_cosine_loss,
    _integrate_with_grad,
    _load_checkpoint_model,
    _sample_latent_batch,
    _style_latent_index,
    _tv_loss,
)
from run_style_embedding_mainline_calibration import (  # noqa: E402
    RECIPES,
    _apply_style_adapter,
    _highpass,
    _l2_mean,
    _resolve_latent_root,
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


def _recipe_by_name(name: str):
    for recipe in RECIPES:
        if recipe.name == name:
            return recipe
    raise ValueError(f"Unknown recipe {name!r}")


def _mean(values: list[float]) -> float:
    return sum(values) / max(1, len(values))


def _row_norm(tensor: torch.Tensor | None, style_id: int) -> float:
    if tensor is None:
        return 0.0
    return float(tensor.detach().float()[int(style_id)].norm().cpu().item())


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Audit whether a frozen-backbone tokenizer objective actually sends "
            "gradient into each style vocabulary row."
        )
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--style-adapter", type=Path, default=None, help="Adapter applied to the student before auditing.")
    parser.add_argument("--teacher-adapter", type=Path, default=None, help="Optional adapter applied to the teacher anchor.")
    parser.add_argument("--latent-root", type=Path, default=None)
    parser.add_argument("--style-subdirs", type=str, default="photo,Hayao,monet,vangogh,cezanne")
    parser.add_argument("--target-style-ids", type=str, default="1,2,3,4")
    parser.add_argument("--recipe", type=str, default="m10_token_vocab_swd_anchor")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-batches", type=int, default=4)
    parser.add_argument("--seed", type=int, default=2718)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "exp/diagnostics/style_tokenizer_gradients")
    args = parser.parse_args()

    recipe = _recipe_by_name(args.recipe)
    style_names = [item.strip() for item in args.style_subdirs.split(",") if item.strip()]
    target_style_ids = [int(item.strip()) for item in args.target_style_ids.split(",") if item.strip()]
    rng = random.Random(int(args.seed))

    model, config = _load_checkpoint_model(args.checkpoint, args.device)
    teacher, _ = _load_checkpoint_model(args.checkpoint, args.device)
    if args.style_adapter is not None:
        _apply_style_adapter(model, args.style_adapter, args.device)
    if args.teacher_adapter is not None:
        _apply_style_adapter(teacher, args.teacher_adapter, args.device)

    tokenizer = getattr(model, "style_tokenizer", None)
    if tokenizer is None:
        raise RuntimeError("checkpoint has no style_tokenizer")

    model.train()
    teacher.eval()
    for param in model.parameters():
        param.requires_grad_(False)
    for param in teacher.parameters():
        param.requires_grad_(False)
    tokenizer.grammar_vocab.weight.requires_grad_(True)
    tokenizer.band_vocab.weight.requires_grad_(True)

    latent_root = _resolve_latent_root(config, args.latent_root)
    latent_index = _style_latent_index(latent_root, style_names)
    content_pool = [p for style in style_names for p in latent_index[style]]
    transport = SWDTransportCost(config)

    batch_rows: list[dict[str, Any]] = []
    summary_buckets: dict[int, dict[str, list[float]]] = {
        style_id: {
            "loss": [],
            "swd": [],
            "anchor": [],
            "grad": [],
            "tv": [],
            "grammar_grad_norm": [],
            "band_grad_norm": [],
        }
        for style_id in target_style_ids
    }

    for style_id in target_style_ids:
        style_name = style_names[style_id]
        for batch_idx in range(1, max(1, int(args.num_batches)) + 1):
            content = _sample_latent_batch(content_pool, max(1, int(args.batch_size)), args.device, rng)
            target = _sample_latent_batch(latent_index[style_name], max(1, int(args.batch_size)), args.device, rng)
            sid = torch.full((content.shape[0],), int(style_id), dtype=torch.long, device=args.device)

            model.zero_grad(set_to_none=True)
            pred = _integrate_with_grad(model, content, style_id=sid, num_steps=recipe.ode_steps)
            with torch.no_grad():
                teacher_pred = _integrate_with_grad(teacher, content, style_id=sid, num_steps=recipe.ode_steps)

            pred_for_swd = _highpass(pred, recipe.highpass_kernel)
            target_for_swd = _highpass(target, recipe.highpass_kernel)
            swd = transport.aligned_cost(pred_for_swd, target_for_swd)
            anchor = _l2_mean(pred, teacher_pred)
            grad_term = _gradient_cosine_loss(pred, content) if recipe.grad_weight > 0.0 else pred.new_tensor(0.0)
            tv = _tv_loss(pred - content) if recipe.delta_tv_weight > 0.0 else pred.new_tensor(0.0)
            loss = (
                recipe.swd_weight * swd
                + recipe.anchor_weight * anchor
                + recipe.grad_weight * grad_term
                + recipe.delta_tv_weight * tv
            )
            if not torch.isfinite(loss.detach()):
                raise FloatingPointError(f"non-finite loss for style={style_name} batch={batch_idx}")
            loss.backward()

            grammar_grad = tokenizer.grammar_vocab.weight.grad
            band_grad = tokenizer.band_vocab.weight.grad
            row = {
                "recipe": recipe.name,
                "style_id": style_id,
                "style": style_name,
                "batch": batch_idx,
                "loss": float(loss.detach().cpu().item()),
                "swd": float(swd.detach().cpu().item()),
                "anchor": float(anchor.detach().cpu().item()),
                "grad": float(grad_term.detach().cpu().item()),
                "tv": float(tv.detach().cpu().item()),
                "grammar_row_norm": _row_norm(tokenizer.grammar_vocab.weight, style_id),
                "band_row_norm": _row_norm(tokenizer.band_vocab.weight, style_id),
                "grammar_grad_norm": _row_norm(grammar_grad, style_id),
                "band_grad_norm": _row_norm(band_grad, style_id),
            }
            batch_rows.append(row)
            for key, values in summary_buckets[style_id].items():
                values.append(float(row[key]))

            del content, target, sid, pred, teacher_pred, loss
            if args.device.startswith("cuda"):
                torch.cuda.empty_cache()

    summary_rows: list[dict[str, Any]] = []
    for style_id in target_style_ids:
        style_name = style_names[style_id]
        bucket = summary_buckets[style_id]
        row = {
            "recipe": recipe.name,
            "style_id": style_id,
            "style": style_name,
            "num_batches": len(bucket["loss"]),
            "mean_loss": _mean(bucket["loss"]),
            "mean_swd": _mean(bucket["swd"]),
            "mean_anchor": _mean(bucket["anchor"]),
            "mean_grad": _mean(bucket["grad"]),
            "mean_tv": _mean(bucket["tv"]),
            "mean_grammar_grad_norm": _mean(bucket["grammar_grad_norm"]),
            "mean_band_grad_norm": _mean(bucket["band_grad_norm"]),
            "current_grammar_row_norm": _row_norm(tokenizer.grammar_vocab.weight, style_id),
            "current_band_row_norm": _row_norm(tokenizer.band_vocab.weight, style_id),
        }
        row["grammar_grad_to_row_norm"] = row["mean_grammar_grad_norm"] / max(row["current_grammar_row_norm"], 1e-8)
        row["band_grad_to_row_norm"] = row["mean_band_grad_norm"] / max(row["current_band_row_norm"], 1e-8)
        summary_rows.append(row)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(args.out_dir / "style_tokenizer_gradient_batches.csv", batch_rows)
    _write_csv(args.out_dir / "style_tokenizer_gradient_summary.csv", summary_rows)
    (args.out_dir / "manifest.json").write_text(
        json.dumps(
            {
                "checkpoint": str(args.checkpoint),
                "style_adapter": str(args.style_adapter or ""),
                "teacher_adapter": str(args.teacher_adapter or ""),
                "latent_root": str(latent_root),
                "recipe": recipe.name,
                "target_style_ids": target_style_ids,
                "batch_size": int(args.batch_size),
                "num_batches": int(args.num_batches),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(args.out_dir)


if __name__ == "__main__":
    main()
