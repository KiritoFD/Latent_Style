from __future__ import annotations

"""Train a small router-aware actuator phase.

The adapter-only memory-bank probes showed that prototype sources are not useful
when collapsed through the frozen style-map actuator. This launcher keeps the
tokenizer/prototype bank fixed, loads it during training, and only trains the
actuator/backbone path that must learn to consume the routed source.
"""

import argparse
import csv
import json
import subprocess
import sys
import time
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from run_style_embedding_distill import _memory_tier_eval_batch_size, _run_full_eval  # noqa: E402


@dataclass(frozen=True)
class RouterBackboneRecipe:
    name: str
    style_adapter: str
    epochs: int
    batch_size: int
    lr: float
    min_lr: float
    trainable_patterns: tuple[str, ...]
    note: str


RECIPES = [
    RouterBackboneRecipe(
        name="ra00_route_actuator_s45_e2",
        style_adapter="exp/style_memory_bank_adapter_route_probe/br00_route_hightex_k4_s45/style_adapter.pt",
        epochs=2,
        batch_size=32,
        lr=5.0e-5,
        min_lr=1.0e-5,
        trainable_patterns=("body_blocks", "blender", "decoder_blocks", "output_head"),
        note="Train only the actuator/body path to read a fixed local prototype router.",
    ),
    RouterBackboneRecipe(
        name="ra01_route_actuator_s65_e2",
        style_adapter="exp/style_memory_bank_adapter_route_probe/br01_route_hightex_k4_s65/style_adapter.pt",
        epochs=2,
        batch_size=32,
        lr=5.0e-5,
        min_lr=1.0e-5,
        trainable_patterns=("body_blocks", "blender", "decoder_blocks", "output_head"),
        note="Same actuator training with the stronger fixed router source.",
    ),
    RouterBackboneRecipe(
        name="rs00_memory_residual_s22_e2",
        style_adapter="exp/style_memory_residual_adapter_probe/mr00_residual_hightex_k4_s22/style_adapter.pt",
        epochs=2,
        batch_size=32,
        lr=4.0e-5,
        min_lr=8.0e-6,
        trainable_patterns=("body_blocks", "blender", "skip_fusion", "decoder_blocks", "dec_post", "dec_mod", "output_head"),
        note="Prototype source bypasses the frozen style-map interface as an explicit body residual; downstream actuator learns to use it.",
    ),
    RouterBackboneRecipe(
        name="rs01_memory_residual_hp_s32_e2",
        style_adapter="exp/style_memory_residual_adapter_probe/mr01_residual_hightex_k4_hp_s32/style_adapter.pt",
        epochs=2,
        batch_size=32,
        lr=4.0e-5,
        min_lr=8.0e-6,
        trainable_patterns=("body_blocks", "blender", "skip_fusion", "decoder_blocks", "dec_post", "dec_mod", "output_head"),
        note="Highpass-gated prototype residual tests whether texton energy can rise without color fog or global style-map collapse.",
    ),
    RouterBackboneRecipe(
        name="rs02_memory_contentdir_s18_e2",
        style_adapter="exp/style_memory_residual_adapter_probe/mr02_residual_contentdir_k4_s18/style_adapter.pt",
        epochs=2,
        batch_size=32,
        lr=4.0e-5,
        min_lr=8.0e-6,
        trainable_patterns=("body_blocks", "blender", "skip_fusion", "decoder_blocks", "dec_post", "dec_mod", "output_head"),
        note="Prototype residual is a local displacement from normalized content feature to target prototype, not an absolute style source.",
    ),
    RouterBackboneRecipe(
        name="rs03_memory_contentdir_hp_s24_e2",
        style_adapter="exp/style_memory_residual_adapter_probe/mr03_residual_contentdir_k4_hp_s24/style_adapter.pt",
        epochs=2,
        batch_size=32,
        lr=4.0e-5,
        min_lr=8.0e-6,
        trainable_patterns=("body_blocks", "blender", "skip_fusion", "decoder_blocks", "dec_post", "dec_mod", "output_head"),
        note="Highpass-gated content-relative prototype displacement tests whether style source needs a transport direction and texton support.",
    ),
    RouterBackboneRecipe(
        name="rt00_typed_fet_s18_e2",
        style_adapter="exp/style_memory_typed_adapter_probe/mt00_typed_fet_k6_s18/style_adapter.pt",
        epochs=2,
        batch_size=32,
        lr=4.0e-5,
        min_lr=8.0e-6,
        trainable_patterns=("body_blocks", "blender", "skip_fusion", "decoder_blocks", "dec_post", "dec_mod", "output_head"),
        note="Typed flat/edge/texton prototypes gate memory displacement by local content support, preventing untyped residual averaging.",
    ),
    RouterBackboneRecipe(
        name="rt01_typed_fet_hp_s24_e2",
        style_adapter="exp/style_memory_typed_adapter_probe/mt01_typed_fet_k6_hp_s24/style_adapter.pt",
        epochs=2,
        batch_size=32,
        lr=4.0e-5,
        min_lr=8.0e-6,
        trainable_patterns=("body_blocks", "blender", "skip_fusion", "decoder_blocks", "dec_post", "dec_mod", "output_head"),
        note="Highpass-gated typed prototype residual tests whether role-aware texton source can raise style without fog.",
    ),
    RouterBackboneRecipe(
        name="rt02_typed_uniform_s20_e2",
        style_adapter="exp/style_memory_typed_adapter_probe/mt02_typed_uniform_fet_k6_s20/style_adapter.pt",
        epochs=2,
        batch_size=32,
        lr=4.0e-5,
        min_lr=8.0e-6,
        trainable_patterns=("body_blocks", "blender", "skip_fusion", "decoder_blocks", "dec_post", "dec_mod", "output_head"),
        note="Typed prototypes use local role gates but near-uniform within-type texture sampling, avoiding content-nearest style averaging.",
    ),
    RouterBackboneRecipe(
        name="rt03_typed_uniform_hp_s24_e2",
        style_adapter="exp/style_memory_typed_adapter_probe/mt03_typed_uniform_fet_k6_hp_s24/style_adapter.pt",
        epochs=2,
        batch_size=32,
        lr=4.0e-5,
        min_lr=8.0e-6,
        trainable_patterns=("body_blocks", "blender", "skip_fusion", "decoder_blocks", "dec_post", "dec_mod", "output_head"),
        note="Highpass-gated version of uniform typed prototype sampling tests whether type-wise texture distribution can lift style cleanly.",
    ),
    RouterBackboneRecipe(
        name="rx00_stylepure_s20_e2",
        style_adapter="exp/style_measure_aligned_adapter_probe/ma00_stylepure_k6_s20/style_adapter.pt",
        epochs=2,
        batch_size=32,
        lr=4.0e-5,
        min_lr=8.0e-6,
        trainable_patterns=("body_blocks", "blender", "skip_fusion", "decoder_blocks", "dec_post", "dec_mod", "output_head"),
        note="Prototype atoms are selected by internal style-purity: close to their target style centroid and far from other style centroids.",
    ),
    RouterBackboneRecipe(
        name="rx01_typed_stylepure_s22_e2",
        style_adapter="exp/style_measure_aligned_adapter_probe/ma01_typed_stylepure_k9_s22/style_adapter.pt",
        epochs=2,
        batch_size=32,
        lr=4.0e-5,
        min_lr=8.0e-6,
        trainable_patterns=("body_blocks", "blender", "skip_fusion", "decoder_blocks", "dec_post", "dec_mod", "output_head"),
        note="Flat/edge/texton atoms are selected by style-purity within each role, so typing is style-measure aligned rather than only content-local.",
    ),
    RouterBackboneRecipe(
        name="rx02_typed_stylepure_uniform_hp_s24_e2",
        style_adapter="exp/style_measure_aligned_adapter_probe/ma02_typed_stylepure_uniform_hp_k9_s24/style_adapter.pt",
        epochs=2,
        batch_size=32,
        lr=4.0e-5,
        min_lr=8.0e-6,
        trainable_patterns=("body_blocks", "blender", "skip_fusion", "decoder_blocks", "dec_post", "dec_mod", "output_head"),
        note="Uniform-within-role style-pure atoms plus highpass support tests whether target-measure atoms help without content-nearest collapse.",
    ),
    RouterBackboneRecipe(
        name="rf00_fisher_stylepure_s20_e2",
        style_adapter="exp/fisher_style_memory_adapter_probe/mf00_fisher_k6_s20/style_adapter.pt",
        epochs=2,
        batch_size=32,
        lr=4.0e-5,
        min_lr=8.0e-6,
        trainable_patterns=("body_blocks", "blender", "skip_fusion", "decoder_blocks", "dec_post", "dec_mod", "output_head"),
        note="Prototype atoms are selected in a Fisher-discriminative internal style space, testing whether raw descriptor inseparability caused style-average residuals.",
    ),
    RouterBackboneRecipe(
        name="rf01_typed_fisher_s22_e2",
        style_adapter="exp/fisher_style_memory_adapter_probe/mf01_typed_fisher_k9_s22/style_adapter.pt",
        epochs=2,
        batch_size=32,
        lr=4.0e-5,
        min_lr=8.0e-6,
        trainable_patterns=("body_blocks", "blender", "skip_fusion", "decoder_blocks", "dec_post", "dec_mod", "output_head"),
        note="Flat/edge/texton atoms are selected after Fisher style projection, combining typed roles with an explicitly separable style metric.",
    ),
]


def _resolve_path(path: Path | str | None) -> Path | None:
    if path is None or str(path).strip() == "":
        return None
    p = Path(path)
    return p if p.is_absolute() else (ROOT / p).resolve()


def _parse_recipes(spec: str) -> list[RouterBackboneRecipe]:
    if not spec.strip():
        return RECIPES
    keep = {item.strip() for item in spec.split(",") if item.strip()}
    selected = [recipe for recipe in RECIPES if recipe.name in keep]
    if not selected:
        raise ValueError(f"No matching router-aware backbone recipes for {spec!r}")
    return selected


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


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
        "hayao_cross_clip_style": hayao_cross.get("clip_style", float("nan")),
        "hayao_cross_content_lpips": hayao_cross.get("content_lpips", float("nan")),
        "weakest_cross_target": weakest[0],
        "weakest_cross_clip_style": weakest[1].get("clip_style", float("nan")),
        "weakest_cross_content_lpips": weakest[1].get("content_lpips", float("nan")),
    }


def _base_epoch(checkpoint: Path) -> int:
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    return int(payload.get("epoch", 0))


def _build_config(
    recipe: RouterBackboneRecipe,
    *,
    base_config: dict,
    checkpoint: Path,
    out_dir: Path,
    max_train_batches: int,
    latent_root: Path | None,
) -> tuple[dict, int]:
    cfg = deepcopy(base_config)
    base_epoch = _base_epoch(checkpoint)
    train = cfg.setdefault("training", {})
    data = cfg.setdefault("data", {})
    ckpt = cfg.setdefault("checkpoint", {})
    train["resume_checkpoint"] = str(checkpoint)
    train["style_adapter_path"] = str(_resolve_path(recipe.style_adapter))
    train["trainable_name_patterns"] = list(recipe.trainable_patterns)
    train["freeze_name_patterns"] = ["style_emb", "style_spatial_id_16", "style_tokenizer"]
    train["num_epochs"] = int(base_epoch + recipe.epochs)
    train["save_interval"] = 1
    train["batch_size"] = int(recipe.batch_size)
    train["learning_rate"] = float(recipe.lr)
    train["min_learning_rate"] = float(recipe.min_lr)
    train["scheduler"] = "cosine"
    train["use_amp"] = False
    train["channels_last"] = False
    train["use_tqdm"] = True
    train["max_train_batches_per_epoch"] = int(max_train_batches)
    train["numeric_debug"] = True
    train["numeric_debug_interval"] = 25
    train["numeric_debug_halt_on_nonfinite"] = True
    train["distill"] = {}
    if latent_root is not None:
        data["data_root"] = str(latent_root)
    ckpt["save_dir"] = str(out_dir.resolve())
    return cfg, base_epoch + recipe.epochs


def _run_recipe(
    recipe: RouterBackboneRecipe,
    *,
    checkpoint: Path,
    out_root: Path,
    eval_batch_size: int,
    vae_model: str,
    max_train_batches: int,
    latent_root: Path | None,
    skip_eval: bool,
) -> dict:
    checkpoint = checkpoint.resolve()
    style_adapter = _resolve_path(recipe.style_adapter)
    if style_adapter is None or not style_adapter.exists():
        raise FileNotFoundError(f"Recipe style adapter missing: {style_adapter}")
    source = torch.load(checkpoint, map_location="cpu", weights_only=False)
    recipe_dir = out_root / recipe.name
    cfg, final_epoch = _build_config(
        recipe,
        base_config=source["config"],
        checkpoint=checkpoint,
        out_dir=recipe_dir,
        max_train_batches=max_train_batches,
        latent_root=latent_root,
    )
    config_path = recipe_dir / "config.json"
    _write_json(config_path, cfg)
    _write_json(
        recipe_dir / "run_manifest.json",
        {
            "recipe": recipe.__dict__,
            "checkpoint": str(checkpoint),
            "style_adapter": str(style_adapter),
            "one_line_hypothesis": recipe.note,
        },
    )
    start = time.time()
    cmd = [sys.executable, str(ROOT / "src" / "run.py"), "--config", str(config_path)]
    subprocess.run(cmd, cwd=str(ROOT), check=True)
    final_ckpt = recipe_dir / f"epoch_{final_epoch:04d}.pt"
    if not final_ckpt.exists():
        raise FileNotFoundError(f"Expected final checkpoint missing: {final_ckpt}")
    row = {
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
        row.update(_read_summary_metrics(summary))
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description="Train/evaluate router-aware backbone probes.")
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "exp/vae_backend/ema_transport_moment/ema_transport_adain_w34_guard/epoch_0006.pt")
    parser.add_argument("--out-root", type=Path, default=ROOT / "exp/router_aware_backbone_probe")
    parser.add_argument("--recipes", default="")
    parser.add_argument("--eval-batch-size", type=int, default=0)
    parser.add_argument("--vae-model", default="auto")
    parser.add_argument("--max-train-batches", type=int, default=0)
    parser.add_argument("--latent-root", type=Path, default=None)
    parser.add_argument("--skip-eval", action="store_true")
    args = parser.parse_args()

    checkpoint = _resolve_path(args.checkpoint)
    out_root = _resolve_path(args.out_root)
    latent_root = _resolve_path(args.latent_root)
    if checkpoint is None or out_root is None:
        raise ValueError("checkpoint and out-root are required")
    recipes = _parse_recipes(args.recipes)
    eval_batch_size = _memory_tier_eval_batch_size("cuda" if torch.cuda.is_available() else "cpu", args.eval_batch_size if args.eval_batch_size > 0 else None)
    out_root.mkdir(parents=True, exist_ok=True)
    rows = [
        _run_recipe(
            recipe,
            checkpoint=checkpoint,
            out_root=out_root,
            eval_batch_size=eval_batch_size,
            vae_model=args.vae_model,
            max_train_batches=max(0, int(args.max_train_batches)),
            latent_root=latent_root,
            skip_eval=bool(args.skip_eval),
        )
        for recipe in recipes
    ]
    fields = sorted({key for row in rows for key in row.keys()})
    with (out_root / "router_aware_backbone_results.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(out_root / "router_aware_backbone_results.csv")


if __name__ == "__main__":
    main()
