from __future__ import annotations

"""Train the backbone/actuator to consume the frozen fo11 tokenizer operator.

fo11 proved that Fisher-initialized depthwise grammar filters are executable,
but tokenizer-only training stayed flat. This launcher starts from the fo11
checkpoint, freezes the tokenizer/style identity source, and trains only the
consumer path. The hypothesis is intentionally narrow:

    fixed operator semantics + trainable actuator consumption can convert the
    measured depthwise delta into target-style transport.
"""

import argparse
import csv
import json
import subprocess
import sys
import time
from copy import deepcopy
from dataclasses import dataclass, asdict
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

from run_style_embedding_distill import _memory_tier_eval_batch_size, _run_full_eval  # noqa: E402


@dataclass(frozen=True)
class ConsumerRecipe:
    name: str
    epochs: int
    batch_size: int
    lr: float
    min_lr: float
    terminal_swd_scale: float
    trainable_patterns: tuple[str, ...]
    note: str
    model_overrides: tuple[tuple[str, Any], ...] = ()
    resume_allow_missing_name_patterns: tuple[str, ...] = ()
    lr_multipliers: tuple[tuple[str, float], ...] = ()
    freeze_patterns: tuple[str, ...] = ("style_emb", "style_spatial_id_16", "style_tokenizer")


RECIPES = [
    ConsumerRecipe(
        name="fo20_learned_style_operator_alphabet_e2",
        epochs=2,
        batch_size=32,
        lr=4.5e-4,
        min_lr=6.0e-5,
        terminal_swd_scale=1.0,
        trainable_patterns=(
            "token_depthwise_filter_style_basis_gate_logits",
            "token_depthwise_filter_style_basis_delta",
            "style_tokenizer.grammar_vocab",
            "style_tokenizer.band_vocab",
        ),
        note=(
            "Resume fo18 and learn a style-local zero-mean high-pass operator alphabet, plus tokenizer grammar/band assignment. "
            "This directly tests the fo18 conclusion that fixed Sobel/Laplace depthwise bases are not style-discriminative enough."
        ),
        model_overrides=(
            ("style_token_depthwise_filter_style_basis_gate", True),
            ("style_token_depthwise_filter_style_basis_gate_scale", 0.75),
            ("style_token_depthwise_filter_style_basis_delta", True),
            ("style_token_depthwise_filter_style_basis_delta_scale", 0.30),
        ),
        resume_allow_missing_name_patterns=("token_depthwise_filter_style_basis_delta",),
        lr_multipliers=(
            ("token_depthwise_filter_style_basis_gate_logits", 3.0),
            ("token_depthwise_filter_style_basis_delta", 1.5),
            ("style_tokenizer.band_vocab", 0.35),
        ),
        freeze_patterns=("style_emb", "style_spatial_id_16"),
    ),
    ConsumerRecipe(
        name="fo18_depthwise_style_basis_gate_e2",
        epochs=2,
        batch_size=32,
        lr=3.0e-3,
        min_lr=5.0e-4,
        terminal_swd_scale=1.0,
        trainable_patterns=("token_depthwise_filter_style_basis_gate_logits",),
        note=(
            "Freeze fo11 tokenizer and backbone; add a style-id by depthwise-basis gate inside the grammar operator. "
            "This tests whether style-local operator basis allocation can move style without output-head repaint."
        ),
        model_overrides=(
            ("style_token_depthwise_filter_style_basis_gate", True),
            ("style_token_depthwise_filter_style_basis_gate_scale", 0.75),
        ),
        resume_allow_missing_name_patterns=("token_depthwise_filter_style_basis_gate_logits",),
    ),
    ConsumerRecipe(
        name="fo17_depthwise_gate_head_e2",
        epochs=2,
        batch_size=32,
        lr=5.0e-5,
        min_lr=7.0e-6,
        terminal_swd_scale=1.0,
        trainable_patterns=("token_depthwise_filter_gate_logits", "output_head"),
        note=(
            "Freeze fo11 tokenizer and backbone body; train the connected depthwise mid/high gate plus output_head. "
            "This tests the smallest useful consumer surface where depthwise transport becomes latent delta."
        ),
        model_overrides=(
            ("style_token_depthwise_filter_learnable_gate", True),
            ("style_token_depthwise_filter_learnable_gate_scale", 0.75),
        ),
        resume_allow_missing_name_patterns=("token_depthwise_filter_gate_logits",),
        lr_multipliers=(("token_depthwise_filter_gate_logits", 50.0), ("output_head", 1.0)),
    ),
    ConsumerRecipe(
        name="fo16_depthwise_gate_only_e2",
        epochs=2,
        batch_size=32,
        lr=3.0e-3,
        min_lr=5.0e-4,
        terminal_swd_scale=1.0,
        trainable_patterns=("token_depthwise_filter_gate_logits",),
        note=(
            "Freeze fo11 tokenizer and backbone; add a two-scalar learnable mid/high gate directly on the "
            "depthwise grammar operator. This is the narrow valid operator actuator after blender-only proved no-grad."
        ),
        model_overrides=(
            ("style_token_depthwise_filter_learnable_gate", True),
            ("style_token_depthwise_filter_learnable_gate_scale", 0.75),
        ),
        resume_allow_missing_name_patterns=("token_depthwise_filter_gate_logits",),
    ),
    ConsumerRecipe(
        name="fo14_depthwise_blender_only_e2",
        epochs=2,
        batch_size=32,
        lr=4.0e-5,
        min_lr=8.0e-6,
        terminal_swd_scale=1.0,
        trainable_patterns=("blender",),
        note=(
            "Freeze the fo11 tokenizer and the transport body/decoder; train only StyleBlender as the narrow "
            "operator interface. This tests whether depthwise grammar can be consumed without fo12-style repaint drift."
        ),
    ),
    ConsumerRecipe(
        name="fo15_depthwise_blender_head_e2",
        epochs=2,
        batch_size=32,
        lr=3.5e-5,
        min_lr=7.0e-6,
        terminal_swd_scale=1.0,
        trainable_patterns=("blender", "output_head"),
        note=(
            "Freeze the fo11 tokenizer and transport body/decoder; train StyleBlender plus the final output head. "
            "This is the smallest capacity increase after blender-only if the interface underfits."
        ),
    ),
    ConsumerRecipe(
        name="fo12_depthwise_consumer_guard_e2",
        epochs=2,
        batch_size=32,
        lr=3.0e-5,
        min_lr=6.0e-6,
        terminal_swd_scale=1.0,
        trainable_patterns=(
            "body_blocks",
            "blender",
            "skip_fusion",
            "decoder_blocks",
            "dec_post",
            "dec_mod",
            "output_head",
        ),
        note=(
            "Freeze fo11 tokenizer/style identity source; train only the body/blender/decoder consumer so the "
            "existing depthwise grammar operator can become a learned style transport path."
        ),
    ),
    ConsumerRecipe(
        name="fo13_depthwise_consumer_style_e2",
        epochs=2,
        batch_size=32,
        lr=3.5e-5,
        min_lr=7.0e-6,
        terminal_swd_scale=1.18,
        trainable_patterns=(
            "body_blocks",
            "blender",
            "skip_fusion",
            "decoder_blocks",
            "dec_post",
            "dec_mod",
            "output_head",
        ),
        note=(
            "Same frozen fo11 tokenizer, with a mild terminal-SWD scale-up. This tests whether the consumer path "
            "needs a little more style gradient once the operator semantics are fixed."
        ),
    ),
]


def _resolve_path(path: Path | str | None) -> Path | None:
    if path is None or str(path).strip() == "":
        return None
    p = Path(path)
    return p if p.is_absolute() else (ROOT / p).resolve()


def _parse_recipes(spec: str) -> list[ConsumerRecipe]:
    if not spec.strip():
        return RECIPES
    keep = {item.strip() for item in spec.split(",") if item.strip()}
    chosen = [recipe for recipe in RECIPES if recipe.name in keep]
    if not chosen:
        raise ValueError(f"No matching Fisher operator consumer recipes for {spec!r}")
    return chosen


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def _base_epoch(checkpoint: Path) -> int:
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    return int(payload.get("epoch", 0))


def _read_summary_metrics(summary: dict[str, Any]) -> dict[str, Any]:
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
        "hayao_cross_clip_style": hayao_cross.get("clip_style", float("nan")),
        "hayao_cross_content_lpips": hayao_cross.get("content_lpips", float("nan")),
        "weakest_cross_target": weakest[0],
        "weakest_cross_clip_style": weakest[1].get("clip_style", float("nan")),
        "weakest_cross_content_lpips": weakest[1].get("content_lpips", float("nan")),
    }


def _build_config(
    recipe: ConsumerRecipe,
    *,
    source_config: dict[str, Any],
    checkpoint: Path,
    style_adapter: Path,
    out_dir: Path,
    latent_root: Path | None,
    max_train_batches: int,
) -> tuple[dict[str, Any], int]:
    cfg = deepcopy(source_config)
    final_epoch = _base_epoch(checkpoint) + int(recipe.epochs)
    train = cfg.setdefault("training", {})
    data = cfg.setdefault("data", {})
    bridge = cfg.setdefault("bridge", {})
    ckpt = cfg.setdefault("checkpoint", {})

    train["resume_checkpoint"] = str(checkpoint)
    train["style_adapter_path"] = str(style_adapter)
    train["resume_skip_optimizer"] = True
    train["resume_allow_missing_name_patterns"] = list(recipe.resume_allow_missing_name_patterns)
    train["trainable_name_patterns"] = list(recipe.trainable_patterns)
    train["freeze_name_patterns"] = list(recipe.freeze_patterns)
    train["num_epochs"] = int(final_epoch)
    train["save_interval"] = 1
    train["batch_size"] = int(recipe.batch_size)
    train["learning_rate"] = float(recipe.lr)
    train["min_learning_rate"] = float(recipe.min_lr)
    train["trainable_lr_multipliers"] = [list(item) for item in recipe.lr_multipliers]
    train["scheduler"] = "cosine"
    train["use_amp"] = False
    train["channels_last"] = False
    train["use_tqdm"] = True
    train["max_train_batches_per_epoch"] = int(max_train_batches)
    train["numeric_debug"] = True
    train["numeric_debug_interval"] = 25
    train["numeric_debug_halt_on_nonfinite"] = True
    train["distill"] = {}

    model = cfg.setdefault("model", {})
    for key, value in recipe.model_overrides:
        model[str(key)] = value
    if "terminal_swd_weight" in bridge:
        bridge["terminal_swd_weight"] = float(bridge["terminal_swd_weight"]) * float(recipe.terminal_swd_scale)
    if latent_root is not None:
        data["data_root"] = str(latent_root)
    ckpt["save_dir"] = str(out_dir.resolve())
    return cfg, final_epoch


def _run_recipe(
    recipe: ConsumerRecipe,
    *,
    checkpoint: Path,
    style_adapter: Path,
    out_root: Path,
    latent_root: Path | None,
    max_train_batches: int,
    eval_batch_size: int,
    vae_model: str,
    skip_eval: bool,
) -> dict[str, Any]:
    checkpoint = checkpoint.resolve()
    style_adapter = style_adapter.resolve()
    if not checkpoint.exists():
        raise FileNotFoundError(f"Missing source checkpoint: {checkpoint}")
    if not style_adapter.exists():
        raise FileNotFoundError(f"Missing source style adapter: {style_adapter}")

    source = torch.load(checkpoint, map_location="cpu", weights_only=False)
    recipe_dir = out_root / recipe.name
    cfg, final_epoch = _build_config(
        recipe,
        source_config=source["config"],
        checkpoint=checkpoint,
        style_adapter=style_adapter,
        out_dir=recipe_dir,
        latent_root=latent_root,
        max_train_batches=max_train_batches,
    )
    config_path = recipe_dir / "config.json"
    _write_json(config_path, cfg)
    _write_json(
        recipe_dir / "run_manifest.json",
        {
            "recipe": asdict(recipe),
            "checkpoint": str(checkpoint),
            "style_adapter": str(style_adapter),
            "model_overrides": dict(recipe.model_overrides),
            "one_line_hypothesis": recipe.note,
            "main_omf_loss_changed": False,
            "tokenizer_policy": (
                "selected tokenizer grammar/band fields trainable"
                if "style_tokenizer" not in recipe.freeze_patterns
                else "frozen fo11 Fisher/depthwise tokenizer fields"
            ),
            "rejection_gate": "reject if clip_style does not beat ag02 or LPIPS/grid worsens materially",
        },
    )

    start = time.time()
    subprocess.run(
        [sys.executable, str(ROOT / "src" / "run.py"), "--config", str(config_path)],
        cwd=str(ROOT),
        check=True,
    )
    final_ckpt = recipe_dir / f"epoch_{final_epoch:04d}.pt"
    if not final_ckpt.exists():
        raise FileNotFoundError(f"Expected final checkpoint missing: {final_ckpt}")

    row: dict[str, Any] = {
        "recipe": recipe.name,
        "checkpoint": str(final_ckpt),
        "style_adapter": str(style_adapter),
        "seconds_train": time.time() - start,
        "mode": "train_only" if skip_eval else "train_eval",
    }
    if not skip_eval:
        summary = _run_full_eval(
            checkpoint=final_ckpt,
            style_adapter=style_adapter,
            output_dir=recipe_dir / "full_eval" / f"epoch_{final_epoch:04d}",
            batch_size=eval_batch_size,
            vae_model=vae_model,
        )
        _write_json(recipe_dir / "full_eval_summary.json", summary)
        row.update(_read_summary_metrics(summary))
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=ROOT / "exp/fisher_operator_tokenizer_probe/fo11_depthwise_filter_swd80/checkpoint_fisher_operator_tokenizer.pt",
    )
    parser.add_argument(
        "--style-adapter",
        type=Path,
        default=ROOT / "exp/fisher_operator_tokenizer_probe/fo11_depthwise_filter_swd80/style_adapter.pt",
    )
    parser.add_argument("--out-root", type=Path, default=ROOT / "exp/fisher_operator_consumer_probe")
    parser.add_argument("--recipes", default="")
    parser.add_argument("--latent-root", type=Path, default=None)
    parser.add_argument("--max-train-batches", type=int, default=0)
    parser.add_argument("--eval-batch-size", type=int, default=0)
    parser.add_argument("--vae-model", default="auto")
    parser.add_argument("--skip-eval", action="store_true")
    args = parser.parse_args()

    checkpoint = _resolve_path(args.checkpoint)
    style_adapter = _resolve_path(args.style_adapter)
    out_root = _resolve_path(args.out_root)
    latent_root = _resolve_path(args.latent_root)
    if checkpoint is None or style_adapter is None or out_root is None:
        raise ValueError("checkpoint, style-adapter, and out-root are required")
    out_root.mkdir(parents=True, exist_ok=True)
    eval_batch_size = _memory_tier_eval_batch_size(
        "cuda" if torch.cuda.is_available() else "cpu",
        args.eval_batch_size if args.eval_batch_size > 0 else None,
    )
    rows = [
        _run_recipe(
            recipe,
            checkpoint=checkpoint,
            style_adapter=style_adapter,
            out_root=out_root,
            latent_root=latent_root,
            max_train_batches=max(0, int(args.max_train_batches)),
            eval_batch_size=eval_batch_size,
            vae_model=str(args.vae_model),
            skip_eval=bool(args.skip_eval),
        )
        for recipe in _parse_recipes(str(args.recipes))
    ]
    fields = sorted({key for row in rows for key in row.keys()})
    with (out_root / "fisher_operator_consumer_results.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(out_root / "fisher_operator_consumer_results.csv")


if __name__ == "__main__":
    main()
