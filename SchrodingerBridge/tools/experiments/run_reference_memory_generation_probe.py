from __future__ import annotations

"""Reference-memory generation probe for the m02/ag02 plateau.

This is a diagnostic, not a new training objective. It keeps the checkpoint and
style adapter frozen, then replaces the id-only spatial source at inference with
target-style latents drawn from the internal training latent pools. The goal is
to test whether the current style plateau is caused by the learned style source
being too centroid-like.
"""

import argparse
import csv
import json
import os
import random
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from run_style_embedding_distill import (  # noqa: E402
    _infer_vae_model,
    _load_latent,
    _memory_tier_eval_batch_size,
    _read_json,
    _resolve_eval_settings,
    _style_latent_index,
)
from utils.inference import LGTInference, decode_latent, encode_image, load_vae  # noqa: E402
from utils.run_evaluation import _load_eval_image_tensor  # noqa: E402


@dataclass(frozen=True)
class ReferenceMemoryRecipe:
    name: str
    mode: str
    max_refs_per_style: int = 96
    candidate_k: int = 8
    highpass_kernel: int = 5


RECIPES = [
    ReferenceMemoryRecipe(
        name="rm00_random_ref1",
        mode="random",
        max_refs_per_style=96,
        candidate_k=1,
    ),
    ReferenceMemoryRecipe(
        name="rm01_lowfreq_match_k8",
        mode="lowfreq_match",
        max_refs_per_style=128,
        candidate_k=8,
    ),
]


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def _parse_recipes(spec: str) -> list[ReferenceMemoryRecipe]:
    if not spec.strip():
        return RECIPES
    keep = {item.strip() for item in spec.split(",") if item.strip()}
    selected = [recipe for recipe in RECIPES if recipe.name in keep]
    if not selected:
        raise ValueError(f"No matching reference-memory recipes for {spec!r}")
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


def _summary_metrics(summary: dict) -> dict:
    overview = dict(summary.get("analysis", {}).get("all_pairs_overview", {}) or {})
    cross = dict(summary.get("analysis", {}).get("cross_by_target_style", {}) or {})
    hayao = dict(cross.get("Hayao", {}) or cross.get("hayao", {}) or {})
    valid_targets = [
        (str(name), dict(payload))
        for name, payload in cross.items()
        if isinstance(payload, dict) and payload.get("clip_style") is not None
    ]
    valid_targets.sort(key=lambda item: float(item[1].get("clip_style", float("inf"))))
    weakest = valid_targets[0] if valid_targets else ("", {})
    return {
        "clip_style": overview.get("clip_style", float("nan")),
        "clip_content": overview.get("clip_content", float("nan")),
        "content_lpips": overview.get("content_lpips", float("nan")),
        "ec": overview.get("edge_consistency", overview.get("ec", float("nan"))),
        "hayao_cross_clip_style": hayao.get("clip_style", float("nan")),
        "hayao_cross_content_lpips": hayao.get("content_lpips", float("nan")),
        "weakest_cross_target": weakest[0],
        "weakest_cross_clip_style": weakest[1].get("clip_style", float("nan")),
        "weakest_cross_content_lpips": weakest[1].get("content_lpips", float("nan")),
    }


def _highpass(x: torch.Tensor, kernel: int) -> torch.Tensor:
    kernel = max(1, int(kernel))
    if kernel <= 1:
        return x.float()
    if kernel % 2 == 0:
        kernel += 1
    return x.float() - F.avg_pool2d(x.float(), kernel_size=kernel, stride=1, padding=kernel // 2)


def _lowpass(x: torch.Tensor, kernel: int) -> torch.Tensor:
    kernel = max(1, int(kernel))
    if kernel <= 1:
        return x.float()
    if kernel % 2 == 0:
        kernel += 1
    return F.avg_pool2d(x.float(), kernel_size=kernel, stride=1, padding=kernel // 2)


def _descriptor(x: torch.Tensor, *, kind: str, kernel: int) -> torch.Tensor:
    feat = _lowpass(x, kernel) if kind == "low" else _highpass(x, kernel)
    flat = feat.flatten(1).float()
    return F.normalize(flat, dim=1, eps=1e-6).cpu()


def _build_reference_bank(
    latent_index: dict[str, list[Path]],
    *,
    recipe: ReferenceMemoryRecipe,
    rng: random.Random,
) -> dict[str, dict[str, object]]:
    bank: dict[str, dict[str, object]] = {}
    for style_name, paths in latent_index.items():
        selected = list(paths)
        rng.shuffle(selected)
        selected = selected[: max(1, int(recipe.max_refs_per_style))]
        latents = torch.cat([_load_latent(path) for path in selected], dim=0).cpu()
        low_desc = _descriptor(latents, kind="low", kernel=recipe.highpass_kernel)
        high_desc = _descriptor(latents, kind="high", kernel=recipe.highpass_kernel)
        high_energy = _highpass(latents, recipe.highpass_kernel).abs().mean(dim=(1, 2, 3)).cpu()
        bank[style_name] = {
            "paths": selected,
            "latents": latents,
            "low_desc": low_desc,
            "high_desc": high_desc,
            "high_energy": high_energy,
        }
    return bank


def _select_reference_latents(
    recipe: ReferenceMemoryRecipe,
    *,
    bank_row: dict[str, object],
    source_latents: torch.Tensor,
    rng: random.Random,
) -> tuple[torch.Tensor, list[int], list[float]]:
    refs = bank_row["latents"]
    if not torch.is_tensor(refs):
        raise TypeError("reference bank latents are missing")
    count = int(refs.shape[0])
    if recipe.mode == "random":
        indices = [rng.randrange(count) for _ in range(source_latents.shape[0])]
        scores = [0.0 for _ in indices]
        return refs.index_select(0, torch.tensor(indices, dtype=torch.long)), indices, scores
    if recipe.mode == "lowfreq_match":
        low_desc = bank_row["low_desc"]
        high_energy = bank_row["high_energy"]
        if not torch.is_tensor(low_desc) or not torch.is_tensor(high_energy):
            raise TypeError("reference bank descriptors are missing")
        k = min(max(1, int(recipe.candidate_k)), count)
        top = torch.topk(high_energy.float(), k=k).indices
        src_desc = _descriptor(source_latents.cpu(), kind="low", kernel=recipe.highpass_kernel)
        sim = src_desc @ low_desc.index_select(0, top).T
        local_idx = sim.argmax(dim=1)
        chosen = top.index_select(0, local_idx)
        scores = sim.max(dim=1).values.detach().cpu().tolist()
        return refs.index_select(0, chosen), chosen.detach().cpu().tolist(), [float(v) for v in scores]
    raise ValueError(f"Unsupported recipe mode: {recipe.mode}")


def _collect_eval_sources(config: dict, checkpoint: Path, max_src_samples: int | None) -> tuple[list[str], Path, list[dict]]:
    settings = _resolve_eval_settings(config)
    style_names = list(settings["style_subdirs"])
    if not style_names:
        raise ValueError("Checkpoint config has no data.style_subdirs")
    test_dir = (checkpoint.parent / str(settings["test_image_dir"])).resolve()
    if not test_dir.exists():
        test_dir = (ROOT / str(settings["test_image_dir"])).resolve()
    if not test_dir.exists():
        raise FileNotFoundError(f"Missing test image dir: {settings['test_image_dir']}")
    cap = int(settings["max_src_samples"] if max_src_samples is None else max_src_samples)
    all_src: list[dict] = []
    for style_id, style_name in enumerate(style_names):
        s_dir = test_dir / style_name
        images = sorted(p for p in s_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"})
        rng = random.Random(42)
        rng.shuffle(images)
        if cap > 0:
            images = images[:cap]
        for path in images:
            all_src.append({"path": path, "style_id": style_id, "style_name": style_name})
    return style_names, test_dir, all_src


@torch.no_grad()
def _generate_recipe(
    recipe: ReferenceMemoryRecipe,
    *,
    checkpoint: Path,
    style_adapter: Path,
    latent_root: Path,
    out_root: Path,
    batch_size: int,
    vae_model: str,
    max_src_samples: int | None,
    seed: int,
    device: str,
    force_integrate: bool,
    skip_eval: bool,
) -> dict:
    rng = random.Random(seed)
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    config = ckpt["config"]
    resolved_vae_model = _infer_vae_model(config, vae_model)
    style_names, test_dir, all_src = _collect_eval_sources(config, checkpoint, max_src_samples)
    latent_index = _style_latent_index(latent_root, style_names)
    bank = _build_reference_bank(latent_index, recipe=recipe, rng=rng)

    recipe_dir = out_root / recipe.name
    images_dir = recipe_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    _write_json(
        recipe_dir / "reference_memory_config.json",
        {
            "recipe": recipe.__dict__,
            "checkpoint": str(checkpoint),
            "style_adapter": str(style_adapter),
            "latent_root": str(latent_root),
            "test_dir": str(test_dir),
            "vae_model": resolved_vae_model,
            "max_src_samples": max_src_samples,
            "batch_size": batch_size,
            "one_line_hypothesis": (
                "If the plateau is caused by a centroid-like id-only style source, "
                "internal target-style reference latents should lift style before any loss change."
            ),
        },
    )

    vae = load_vae(device=device, model_id=resolved_vae_model)
    lgt = LGTInference(
        str(checkpoint.resolve()),
        device=device,
        num_steps=int((_resolve_eval_settings(config))["num_steps"]),
        step_size=float((_resolve_eval_settings(config))["step_size"]),
        style_strength=(_resolve_eval_settings(config))["style_strength"],
        style_adapter_path=str(style_adapter.resolve()),
        force_integrate=force_integrate,
    )
    model_scale = float(getattr(lgt.model, "latent_scale_factor", 0.18215))
    vae_scale = float(getattr(getattr(vae, "config", None), "scaling_factor", model_scale))
    scale_in = model_scale / max(vae_scale, 1e-8)
    scale_out = vae_scale / max(model_scale, 1e-8)

    selection_rows: list[dict] = []
    started = time.time()
    for start in range(0, len(all_src), batch_size):
        batch_info = all_src[start : start + batch_size]
        src_batch = torch.stack([_load_eval_image_tensor(item["path"]) for item in batch_info], dim=0).to(device)
        latents_src = encode_image(vae, src_batch, device).float()
        if abs(scale_in - 1.0) > 1e-4:
            latents_src = latents_src * scale_in
        latents_x0 = lgt.inversion(latents_src)

        for tgt_id, tgt_name in enumerate(style_names):
            ref_cpu, ref_indices, ref_scores = _select_reference_latents(
                recipe,
                bank_row=bank[tgt_name],
                source_latents=latents_x0.detach().cpu(),
                rng=rng,
            )
            ref = ref_cpu.to(device=device, dtype=latents_x0.dtype)
            tgt_ids = torch.full((len(batch_info),), tgt_id, device=device, dtype=torch.long)
            latents_gen = lgt.generation(latents_x0, tgt_ids, target_style_latent=ref)
            if abs(scale_out - 1.0) > 1e-4:
                latents_gen = latents_gen * scale_out
            imgs = decode_latent(vae, latents_gen, device).cpu()
            ref_paths = bank[tgt_name]["paths"]
            for idx, src_item in enumerate(batch_info):
                out_name = f"{src_item['style_name']}_{src_item['path'].stem}_to_{tgt_name}.jpg"
                arr = (imgs[idx].permute(1, 2, 0).numpy() * 255.0).clip(0, 255).astype(np.uint8)
                Image.fromarray(arr).save(images_dir / out_name, quality=95)
                selection_rows.append(
                    {
                        "src_style": src_item["style_name"],
                        "src_name": src_item["path"].stem,
                        "target_style": tgt_name,
                        "ref_index": int(ref_indices[idx]),
                        "ref_path": str(ref_paths[int(ref_indices[idx])]),
                        "selection_score": float(ref_scores[idx]),
                    }
                )
        if str(device).startswith("cuda"):
            torch.cuda.empty_cache()
        print(f"[{recipe.name}] generated {min(start + batch_size, len(all_src))}/{len(all_src)} source rows")

    with (recipe_dir / "selection_rows.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["src_style", "src_name", "target_style", "ref_index", "ref_path", "selection_score"],
        )
        writer.writeheader()
        writer.writerows(selection_rows)

    del lgt, vae
    if str(device).startswith("cuda"):
        torch.cuda.empty_cache()

    if skip_eval:
        summary = {
            "recipe": recipe.name,
            "mode": "generation_only",
            "generated_count": int(len(selection_rows)),
            "seconds": time.time() - started,
        }
        _write_json(recipe_dir / "summary_generation_only.json", summary)
        return summary

    env = dict(os.environ)
    prev_pythonpath = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = str(SRC) if not prev_pythonpath else str(SRC) + os.pathsep + prev_pythonpath
    eval_cmd = [
        sys.executable,
        "-m",
        "utils.run_evaluation",
        str(recipe_dir.resolve()),
        "--output",
        str(recipe_dir.resolve()),
        "--reuse_generated",
        "--eval_lpips_chunk_size",
        "2",
        "--style_subdirs",
        ",".join(style_names),
        "--vae_model",
        resolved_vae_model,
    ]
    subprocess.run(eval_cmd, cwd=str(SRC), env=env, check=True)
    summary_path = recipe_dir / "summary_reuse_generated.json"
    if not summary_path.exists():
        summary_path = recipe_dir / "summary.json"
    summary = _read_json(summary_path)
    metrics = _summary_metrics(summary)
    metrics.update({"recipe": recipe.name, "seconds": time.time() - started})
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Run internal reference-memory generation probes.")
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "exp/vae_backend/ema_transport_moment/ema_transport_adain_w34_guard/epoch_0006.pt")
    parser.add_argument("--style-adapter", type=Path, default=ROOT / "exp/style_embedding_mainline_calibration/ema_transport_adain_w34_e6_fulltrain/m02_embspatial_highpass_style/style_adapter.pt")
    parser.add_argument("--latent-root", type=Path, default=None)
    parser.add_argument("--out-root", type=Path, default=ROOT / "exp/reference_memory_generation_probe")
    parser.add_argument("--recipes", default="")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--eval-batch-size", type=int, default=0)
    parser.add_argument("--max-src-samples", type=int, default=0, help="0 means use config full_eval cap.")
    parser.add_argument("--vae-model", default="auto")
    parser.add_argument("--seed", type=int, default=20260528)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--force-integrate", action="store_true")
    parser.add_argument("--skip-eval", action="store_true")
    args = parser.parse_args()

    checkpoint = _resolve_path(args.checkpoint)
    style_adapter = _resolve_path(args.style_adapter)
    out_root = _resolve_path(args.out_root)
    if checkpoint is None or style_adapter is None or out_root is None:
        raise ValueError("checkpoint/style-adapter/out-root are required")
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    latent_root = _resolve_latent_root(ckpt["config"], args.latent_root)
    recipes = _parse_recipes(args.recipes)
    eval_batch_size = _memory_tier_eval_batch_size(args.device, args.eval_batch_size if args.eval_batch_size > 0 else None)
    batch_size = max(1, int(args.batch_size if args.batch_size > 0 else eval_batch_size))
    max_src_samples = None if int(args.max_src_samples) <= 0 else int(args.max_src_samples)

    rows: list[dict] = []
    out_root.mkdir(parents=True, exist_ok=True)
    for recipe in recipes:
        row = _generate_recipe(
            recipe,
            checkpoint=checkpoint,
            style_adapter=style_adapter,
            latent_root=latent_root,
            out_root=out_root,
            batch_size=batch_size,
            vae_model=args.vae_model,
            max_src_samples=max_src_samples,
            seed=args.seed,
            device=args.device,
            force_integrate=bool(args.force_integrate),
            skip_eval=bool(args.skip_eval),
        )
        rows.append(row)

    fieldnames = sorted({key for row in rows for key in row.keys()})
    with (out_root / "reference_memory_generation_results.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(out_root / "reference_memory_generation_results.csv")


if __name__ == "__main__":
    main()
