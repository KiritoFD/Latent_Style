from __future__ import annotations

"""Probe whether m02 is limited by its learned id-only style source.

The script does not change the backbone or the training objective. It builds a
style-memory adapter by replacing part of `style_spatial_id_16` with body-level
features extracted from the target style latent pools, then evaluates the
result through the normal full-eval path.
"""

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

from run_style_embedding_distill import (  # noqa: E402
    _load_checkpoint_model,
    _load_latent,
    _memory_tier_eval_batch_size,
    _run_full_eval,
    _save_style_adapter,
    _style_latent_index,
)
from run_style_embedding_mainline_calibration import _apply_style_adapter  # noqa: E402


@dataclass(frozen=True)
class MemoryBankRecipe:
    name: str
    mode: str
    blend: float
    highpass_boost: float = 1.0
    highpass_kernel: int = 3
    max_samples_per_style: int = 96
    batch_size: int = 16


RECIPES = [
    MemoryBankRecipe(
        name="mb00_body_mean_blend25",
        mode="mean",
        blend=0.25,
        highpass_boost=1.0,
    ),
    MemoryBankRecipe(
        name="mb01_body_mean_blend50",
        mode="mean",
        blend=0.50,
        highpass_boost=1.15,
    ),
    MemoryBankRecipe(
        name="mb02_body_exemplar_blend35",
        mode="exemplar_high",
        blend=0.35,
        highpass_boost=1.0,
    ),
]


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def _parse_recipes(spec: str) -> list[MemoryBankRecipe]:
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


def _highpass_2d(x: torch.Tensor, kernel: int) -> torch.Tensor:
    kernel = max(1, int(kernel))
    if kernel <= 1:
        return x.float()
    if kernel % 2 == 0:
        kernel += 1
    pad = kernel // 2
    low = F.avg_pool2d(F.pad(x.float(), (pad, pad, pad, pad), mode="reflect"), kernel_size=kernel, stride=1)
    return x.float() - low


def _boost_body_texture(model, proto: torch.Tensor, *, highpass_boost: float, highpass_kernel: int) -> torch.Tensor:
    if abs(float(highpass_boost) - 1.0) < 1e-6:
        return model._normalize_style_map(proto)
    high = _highpass_2d(proto, highpass_kernel)
    low = proto.float() - high
    return model._normalize_style_map(low + high * float(highpass_boost))


@torch.inference_mode()
def _encode_body_features(model, latents: torch.Tensor, style_id: int, device: str) -> torch.Tensor:
    latents = latents.to(device=device, dtype=next(model.parameters()).dtype)
    sid = torch.full((latents.shape[0],), int(style_id), dtype=torch.long, device=latents.device)
    style_code = model.encode_style_id(sid)
    feat = latents / max(float(getattr(model, "latent_scale_factor", 0.18215)), 1e-8)
    h = model.enc_in_act(model.enc_in(feat))
    h = model._run_style_blocks(
        h,
        blocks=model.hires_body,
        style_code=style_code,
        base_idx=0,
        gate_scale=0.0,
    )
    body = model.down(h)
    return model._normalize_style_map(body)


@torch.inference_mode()
def _build_style_prototype(
    model,
    paths: list[Path],
    *,
    style_id: int,
    recipe: MemoryBankRecipe,
    device: str,
    rng: random.Random,
) -> tuple[torch.Tensor, dict]:
    sample_count = min(max(1, int(recipe.max_samples_per_style)), len(paths))
    selected = list(paths)
    rng.shuffle(selected)
    selected = selected[:sample_count]
    batch_size = max(1, int(recipe.batch_size))
    total: torch.Tensor | None = None
    count = 0
    best_feat: torch.Tensor | None = None
    best_score = float("-inf")
    scores: list[float] = []

    for start in range(0, len(selected), batch_size):
        batch_paths = selected[start : start + batch_size]
        latents = torch.cat([_load_latent(path) for path in batch_paths], dim=0)
        feats = _encode_body_features(model, latents, style_id=style_id, device=device).float()
        if recipe.mode == "mean":
            summed = feats.sum(dim=0, keepdim=True)
            total = summed if total is None else total + summed
            count += feats.shape[0]
        elif recipe.mode == "exemplar_high":
            high_energy = _highpass_2d(feats, recipe.highpass_kernel).abs().mean(dim=(1, 2, 3))
            for idx, score in enumerate(high_energy.detach().cpu().tolist()):
                scores.append(float(score))
                if score > best_score:
                    best_score = float(score)
                    best_feat = feats[idx : idx + 1].detach().clone()
        else:
            raise ValueError(f"Unsupported prototype mode: {recipe.mode}")

    if recipe.mode == "mean":
        if total is None or count <= 0:
            raise RuntimeError("No features were accumulated for mean prototype")
        proto = total / float(count)
        score_value = float(_highpass_2d(proto, recipe.highpass_kernel).abs().mean().detach().cpu().item())
    else:
        if best_feat is None:
            raise RuntimeError("No exemplar prototype was selected")
        proto = best_feat
        score_value = best_score
    proto = _boost_body_texture(model, proto, highpass_boost=recipe.highpass_boost, highpass_kernel=recipe.highpass_kernel)
    return proto.detach(), {
        "mode": recipe.mode,
        "sample_count": sample_count,
        "selected_count": len(selected),
        "highpass_score": score_value,
        "score_min": min(scores) if scores else score_value,
        "score_max": max(scores) if scores else score_value,
        "score_mean": (sum(scores) / len(scores)) if scores else score_value,
    }


def _run_recipe(
    recipe: MemoryBankRecipe,
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
) -> dict:
    rng = random.Random(seed)
    model, config = _load_checkpoint_model(checkpoint, device)
    _apply_style_adapter(model, init_style_adapter, device)
    model.eval()
    latent_index = _style_latent_index(latent_root, style_names)

    recipe_dir = out_root / recipe.name
    recipe_dir.mkdir(parents=True, exist_ok=True)
    old_spatial = model._normalize_style_map(model.style_spatial_id_16.detach().float())
    new_spatial = old_spatial.clone()
    proto_rows: list[dict] = []
    start_time = time.time()

    with torch.no_grad():
        for style_id in target_style_ids:
            style_name = style_names[style_id]
            proto, stats = _build_style_prototype(
                model,
                latent_index[style_name],
                style_id=style_id,
                recipe=recipe,
                device=device,
                rng=rng,
            )
            old = old_spatial[style_id : style_id + 1].to(device=proto.device, dtype=proto.dtype)
            blended = model._normalize_style_map(old * (1.0 - float(recipe.blend)) + proto * float(recipe.blend))
            new_spatial[style_id : style_id + 1] = blended.to(device=new_spatial.device, dtype=new_spatial.dtype)
            proto_rows.append(
                {
                    "recipe": recipe.name,
                    "style_id": style_id,
                    "style_name": style_name,
                    "blend": recipe.blend,
                    "highpass_boost": recipe.highpass_boost,
                    **stats,
                    "old_spatial_hp": float(_highpass_2d(old, recipe.highpass_kernel).abs().mean().cpu().item()),
                    "new_spatial_hp": float(_highpass_2d(blended, recipe.highpass_kernel).abs().mean().cpu().item()),
                    "prototype_l2_to_old": float((proto.cpu() - old.cpu()).square().mean().sqrt().item()),
                }
            )
        model.style_spatial_id_16.copy_(new_spatial.to(device=model.style_spatial_id_16.device, dtype=model.style_spatial_id_16.dtype))

    adapter_path = recipe_dir / "style_adapter.pt"
    _save_style_adapter(adapter_path, model)
    with (recipe_dir / "prototype_rows.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(proto_rows[0].keys()))
        writer.writeheader()
        writer.writerows(proto_rows)
    _write_json(
        recipe_dir / "prototype_config.json",
        {
            "recipe": recipe.__dict__,
            "checkpoint": str(checkpoint),
            "init_style_adapter": str(init_style_adapter),
            "latent_root": str(latent_root),
            "style_names": style_names,
            "target_style_ids": target_style_ids,
            "elapsed_seconds": time.time() - start_time,
            "hypothesis": (
                "Replace part of the m02 id-only spatial style source with body-level prototypes "
                "computed from the internal target-style latent pools. If style improves without "
                "semantic imprinting, the bottleneck is style source quality rather than tokenizer amplitude."
            ),
        },
    )

    if skip_eval:
        return {"recipe": recipe.name, "adapter_path": str(adapter_path)}

    full_eval_dir = recipe_dir / "full_eval"
    summary = _run_full_eval(
        checkpoint=checkpoint,
        style_adapter=adapter_path,
        output_dir=full_eval_dir,
        batch_size=eval_batch_size,
        vae_model=vae_model,
    )
    _write_json(recipe_dir / "full_eval_summary.json", summary)
    return {
        "recipe": recipe.name,
        "adapter_path": str(adapter_path),
        "full_eval_dir": str(full_eval_dir),
        **_read_summary_metrics(summary),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build/evaluate data-derived style-memory adapters on the m02 anchor.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--init-style-adapter", type=Path, required=True)
    parser.add_argument("--latent-root", type=Path, default=None)
    parser.add_argument("--out-root", type=Path, default=ROOT / "exp/style_memory_bank_probe")
    parser.add_argument("--style-subdirs", type=str, default="photo,Hayao,monet,vangogh,cezanne")
    parser.add_argument("--target-style-ids", type=str, default="1,2,3,4")
    parser.add_argument("--recipes", type=str, default="")
    parser.add_argument("--eval-batch-size", type=int, default=0)
    parser.add_argument("--vae-model", type=str, default="auto")
    parser.add_argument("--seed", type=int, default=123)
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
    target_style_ids = [int(item.strip()) for item in args.target_style_ids.split(",") if item.strip()]
    recipes = _parse_recipes(args.recipes)
    eval_batch_size = _memory_tier_eval_batch_size(args.device, args.eval_batch_size if args.eval_batch_size > 0 else None)

    args.out_root.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    for idx, recipe in enumerate(recipes):
        rows.append(
            _run_recipe(
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
            )
        )
        with (args.out_root / "style_memory_bank_results.csv").open("w", encoding="utf-8", newline="") as f:
            fieldnames: list[str] = []
            for row in rows:
                for key in row:
                    if key not in fieldnames:
                        fieldnames.append(key)
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
    _write_json(args.out_root / "run_manifest.json", {"checkpoint": str(checkpoint), "rows": rows})
    print(args.out_root / "style_memory_bank_results.csv")


if __name__ == "__main__":
    main()
