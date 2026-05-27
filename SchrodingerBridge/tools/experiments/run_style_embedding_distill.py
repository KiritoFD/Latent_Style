from __future__ import annotations

import argparse
import csv
import json
import math
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

from model import build_model_from_config  # noqa: E402
from ot_cost import SWDTransportCost  # noqa: E402
from config_schema import load_inference_defaults, resolve_full_eval_section  # noqa: E402
from utils.diffeomorphic import _texture_tangent_warp  # noqa: E402
from utils.inference import LGTInference, decode_latent, encode_image, load_vae  # noqa: E402


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def _read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _load_checkpoint_model(checkpoint: Path, device: str):
    ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
    config = ckpt["config"]
    model = build_model_from_config(config["model"], use_checkpointing=False).to(device)
    state = ckpt["model_state_dict"]
    if any(k.startswith("_orig_mod.") for k in state):
        state = {k.replace("_orig_mod.", ""): v for k, v in state.items()}
    model.load_state_dict(state, strict=True)
    return model, config


def _load_latent(path: Path) -> torch.Tensor:
    obj = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(obj, dict):
        for key in ("latent", "z", "image", "tensor"):
            if key in obj:
                obj = obj[key]
                break
    if not torch.is_tensor(obj):
        raise TypeError(f"Unsupported latent payload: {path}")
    if obj.ndim == 3:
        obj = obj.unsqueeze(0)
    return obj.float()


def _style_latent_index(latent_root: Path, style_names: list[str]) -> dict[str, list[Path]]:
    index: dict[str, list[Path]] = {}
    for style in style_names:
        paths = sorted((latent_root / style).glob("*.pt"))
        if not paths:
            raise FileNotFoundError(f"No latent files found for style {style}: {latent_root / style}")
        index[style] = paths
    return index


def _sample_latent_batch(paths: list[Path], batch_size: int, device: str, rng: random.Random) -> torch.Tensor:
    picks = [paths[rng.randrange(len(paths))] for _ in range(batch_size)]
    batch = torch.cat([_load_latent(p) for p in picks], dim=0)
    return batch.to(device)


def _sobel_xy(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    channels = int(x.shape[1])
    kx = x.new_tensor([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]]).view(1, 1, 3, 3)
    ky = x.new_tensor([[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]]).view(1, 1, 3, 3)
    kx = kx.expand(channels, 1, 3, 3).contiguous()
    ky = ky.expand(channels, 1, 3, 3).contiguous()
    gx = F.conv2d(x.float(), kx, padding=1, groups=channels)
    gy = F.conv2d(x.float(), ky, padding=1, groups=channels)
    return gx, gy


def _gradient_cosine_loss(pred: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    gx_p, gy_p = _sobel_xy(pred)
    gx_r, gy_r = _sobel_xy(ref)
    grad_p = torch.cat([gx_p, gy_p], dim=1)
    grad_r = torch.cat([gx_r, gy_r], dim=1)
    cos = F.cosine_similarity(grad_p.flatten(1), grad_r.flatten(1), dim=1, eps=1e-6)
    return 1.0 - cos.mean()


def _tv_loss(warp: torch.Tensor) -> torch.Tensor:
    dx = warp[:, :, :, 1:] - warp[:, :, :, :-1]
    dy = warp[:, :, 1:, :] - warp[:, :, :-1, :]
    return dx.abs().mean() + dy.abs().mean()


def _negative_divergence_clamp(warp: torch.Tensor, tau: float) -> torch.Tensor:
    wx = warp[:, 0:1]
    wy = warp[:, 1:2]
    dwx_dx = wx[:, :, :, 1:] - wx[:, :, :, :-1]
    dwy_dy = wy[:, :, 1:, :] - wy[:, :, :-1, :]
    dwx_dx = F.pad(dwx_dx, (0, 1, 0, 0))
    dwy_dy = F.pad(dwy_dy, (0, 0, 0, 1))
    div = dwx_dx + dwy_dy
    return F.relu(-div - tau).square().mean()


def _integrate_with_grad(model, content: torch.Tensor, style_id: torch.Tensor, num_steps: int) -> torch.Tensor:
    steps = max(1, int(num_steps))
    horizon = 1.0
    if hasattr(model, "_resolve_integration_horizon"):
        horizon = float(model._resolve_integration_horizon(step_size=1.0, style_strength=1.0))
    if horizon <= 0.0:
        return content
    h = content
    dt = horizon / float(steps)
    for step in range(steps):
        t = torch.full(
            (content.shape[0],),
            horizon * ((step + 0.5) / float(steps)),
            dtype=content.dtype,
            device=content.device,
        )
        velocity = model(h, t=t, style_id=style_id)
        h = h + velocity * dt
    return h


def _capture_raw_forward(model, x: torch.Tensor, t: torch.Tensor, style_id: torch.Tensor) -> torch.Tensor:
    holder: dict[str, torch.Tensor] = {}

    def hook(_module, _inputs, output):
        holder["raw"] = output

    handle = model.dec_out.register_forward_hook(hook)
    try:
        _ = model.forward(x, t=t, style_id=style_id)
    finally:
        handle.remove()
    if "raw" not in holder:
        raise RuntimeError("Failed to capture raw stroke output")
    return holder["raw"]


def _stroke_delta_and_warp(model, x: torch.Tensor, raw: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    channels = int(x.shape[1])
    color_strength = float(getattr(model, "diffeomorphic_color_strength", 0.85))
    warp_strength = float(getattr(model, "diffeomorphic_warp_strength", 0.08))
    gate_strength = float(getattr(model, "diffeomorphic_texture_gate_strength", 8.0))
    normal_leak = float(getattr(model, "diffeomorphic_normal_leak", 0.0))

    color_delta = torch.tanh(raw[:, :channels]) * color_strength
    spatial_warp = torch.tanh(raw[:, channels : channels + 2]) * warp_strength
    effective_warp = _texture_tangent_warp(
        x=x,
        warp=spatial_warp,
        gate_strength=gate_strength,
        normal_leak=normal_leak,
    )
    b, _, h, w = x.shape
    grid_y, grid_x = torch.meshgrid(
        torch.linspace(-1.0, 1.0, h, device=x.device, dtype=x.dtype),
        torch.linspace(-1.0, 1.0, w, device=x.device, dtype=x.dtype),
        indexing="ij",
    )
    base_grid = torch.stack([grid_x, grid_y], dim=-1).unsqueeze(0).expand(b, -1, -1, -1)
    warped_grid = (base_grid + effective_warp.permute(0, 2, 3, 1)).clamp(-1.2, 1.2)
    x_warped = F.grid_sample(x.float(), warped_grid, align_corners=False, padding_mode="reflection")
    delta = x_warped + color_delta - x.float()
    return delta, effective_warp


@dataclass(frozen=True)
class DistillRecipe:
    name: str
    optimize_spatial: bool
    calibration_iters: int
    ode_steps: int
    lr: float
    batch_size: int
    tv_weight: float
    grad_weight: float
    div_weight: float
    div_tau: float


DEFAULT_RECIPES = [
    DistillRecipe("d00_emb_only_swd_s4_it60", False, 60, 4, 0.01, 12, 0.0, 0.0, 0.0, 0.2),
    DistillRecipe("d01_embspatial_swd_tv_grad_s4_it80", True, 80, 4, 0.01, 12, 10.0, 1.0, 0.0, 0.2),
    DistillRecipe("d02_embspatial_swd_tv_grad_s8_it80", True, 80, 8, 0.01, 12, 10.0, 1.0, 0.0, 0.2),
    DistillRecipe("d03_embspatial_swd_tv_grad_div_s12_it100", True, 100, 12, 0.008, 10, 10.0, 1.0, 2.0, 0.2),
]


def _parse_recipes(spec: str) -> list[DistillRecipe]:
    if not spec.strip():
        return DEFAULT_RECIPES
    keep = {x.strip() for x in spec.split(",") if x.strip()}
    chosen = [r for r in DEFAULT_RECIPES if r.name in keep]
    if not chosen:
        raise ValueError(f"No matching recipes for: {spec}")
    return chosen


def _save_style_adapter(path: Path, model) -> None:
    payload = {
        "style_emb.weight": model.style_emb.weight.detach().cpu(),
        "style_spatial_id_16": model.style_spatial_id_16.detach().cpu(),
    }
    tokenizer = getattr(model, "style_tokenizer", None)
    if tokenizer is not None:
        payload["style_tokenizer.grammar_vocab.weight"] = tokenizer.grammar_vocab.weight.detach().cpu()
        payload["style_tokenizer.band_vocab.weight"] = tokenizer.band_vocab.weight.detach().cpu()
        identity = getattr(tokenizer, "identity_vocab", None)
        if torch.is_tensor(identity):
            payload["style_tokenizer.identity_vocab"] = identity.detach().cpu()
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def _memory_tier_eval_batch_size(device: str, requested: int | None) -> int:
    if requested is not None and requested > 0:
        return int(requested)
    if device.startswith("cuda") and torch.cuda.is_available():
        total_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        if total_gb <= 8.5:
            return 6
        if total_gb <= 12.5:
            return 8
        return 12
    return 4


def _load_eval_image_tensor(path: Path, size: int = 256) -> torch.Tensor:
    image = Image.open(path).convert("RGB").resize((size, size), Image.Resampling.BICUBIC)
    arr = torch.from_numpy(np.array(image)).float() / 255.0
    return arr.permute(2, 0, 1) * 2.0 - 1.0


def _resolve_eval_settings(config: dict) -> dict:
    defaults = load_inference_defaults().get("full_eval", {}) or {}
    resolved = resolve_full_eval_section(config) or {}
    return {
        "num_steps": int(resolved.get("num_steps", defaults.get("num_steps", 12))),
        "step_size": float(resolved.get("step_size", defaults.get("step_size", 1.0))),
        "style_strength": resolved.get("style_strength", defaults.get("style_strength", 1.0)),
        "max_src_samples": int(resolved.get("max_src_samples", defaults.get("max_src_samples", 30))),
        "test_image_dir": str(config.get("training", {}).get("test_image_dir", "../style_data/overfit50")),
        "style_subdirs": list(config.get("data", {}).get("style_subdirs", [])),
    }


def _infer_vae_model(config: dict, requested: str) -> str:
    req = str(requested or "auto").strip()
    if req and req.lower() != "auto":
        return req
    for section_name in ("training", "data", "model"):
        section = config.get(section_name, {}) or {}
        for key in ("vae_model", "vae_model_id", "vae_backend"):
            value = str(section.get(key, "") or "").strip()
            if value:
                return value
    return "sd15"


@torch.no_grad()
def _generate_eval_images(
    checkpoint: Path,
    style_adapter: Path,
    output_dir: Path,
    config: dict,
    batch_size: int,
    device: str,
    vae_model: str,
) -> None:
    settings = _resolve_eval_settings(config)
    style_names = settings["style_subdirs"]
    test_dir = (checkpoint.parent / settings["test_image_dir"]).resolve()
    if not test_dir.exists():
        test_dir = (ROOT / settings["test_image_dir"]).resolve()
    images_dir = output_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    all_src_info: list[dict] = []
    for style_id, style_name in enumerate(style_names):
        s_dir = test_dir / style_name
        images = sorted([p for p in s_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}])
        rng = random.Random(42)
        rng.shuffle(images)
        images = images[: settings["max_src_samples"]]
        for p in images:
            all_src_info.append({"path": p, "style_id": style_id, "style_name": style_name})

    vae = load_vae(device=device, model_id=vae_model)
    lgt = LGTInference(
        str(checkpoint.resolve()),
        device=device,
        num_steps=settings["num_steps"],
        step_size=settings["step_size"],
        style_strength=settings["style_strength"],
        style_adapter_path=str(style_adapter.resolve()),
    )
    model_scale = float(getattr(lgt.model, "latent_scale_factor", 0.18215))
    vae_scale = float(getattr(vae.config, "scaling_factor", model_scale))
    scale_in = model_scale / max(vae_scale, 1e-8)
    scale_out = vae_scale / max(model_scale, 1e-8)

    total = len(all_src_info)
    for start in range(0, total, batch_size):
        end = min(total, start + batch_size)
        batch_info = all_src_info[start:end]
        src_batch = torch.stack([_load_eval_image_tensor(item["path"]) for item in batch_info], dim=0).to(device)
        latents_src = encode_image(vae, src_batch, device).float()
        if abs(scale_in - 1.0) > 1e-4:
            latents_src = latents_src * scale_in
        latents_x0 = lgt.inversion(latents_src)
        for tgt_id, tgt_name in enumerate(style_names):
            tgt_ids = torch.full((len(batch_info),), tgt_id, device=src_batch.device, dtype=torch.long)
            latents_gen = lgt.generation(latents_x0, tgt_ids)
            if abs(scale_out - 1.0) > 1e-4:
                latents_gen = latents_gen * scale_out
            imgs = decode_latent(vae, latents_gen, device).cpu()
            for i, src_item in enumerate(batch_info):
                out_name = f"{src_item['style_name']}_{src_item['path'].stem}_to_{tgt_name}.jpg"
                out_path = images_dir / out_name
                arr = (imgs[i].permute(1, 2, 0).numpy() * 255.0).clip(0, 255).astype(np.uint8)
                Image.fromarray(arr).save(out_path, quality=95)
        if device.startswith("cuda"):
            torch.cuda.empty_cache()


def _run_full_eval(
    checkpoint: Path,
    style_adapter: Path,
    output_dir: Path,
    batch_size: int,
    vae_model: str,
    extra_args: list[str] | None = None,
) -> dict:
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    config = ckpt["config"]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    resolved_vae_model = _infer_vae_model(config, vae_model)
    _generate_eval_images(
        checkpoint=checkpoint,
        style_adapter=style_adapter,
        output_dir=output_dir,
        config=config,
        batch_size=batch_size,
        device=device,
        vae_model=resolved_vae_model,
    )

    env = dict(**os.environ)
    prev_pythonpath = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = str(SRC) if not prev_pythonpath else str(SRC) + ";" + prev_pythonpath
    cmd = [
        sys.executable,
        "-m",
        "utils.run_evaluation",
        str(output_dir.resolve()),
        "--output",
        str(output_dir.resolve()),
        "--reuse_generated",
        "--eval_lpips_chunk_size",
        "2",
        "--style_subdirs",
        ",".join(config.get("data", {}).get("style_subdirs", [])),
        "--vae_model",
        resolved_vae_model,
    ]
    if extra_args:
        cmd.extend(extra_args)
    subprocess.run(cmd, cwd=str(ROOT / "src"), env=env, check=True)
    summary_path = output_dir / "summary.json"
    if not summary_path.exists():
        reuse_summary = output_dir / "summary_reuse_generated.json"
        if reuse_summary.exists():
            summary_path = reuse_summary
    if not summary_path.exists():
        raise FileNotFoundError(f"Missing full_eval summary: {summary_path}")
    return _read_json(summary_path)


def _summary_row(recipe: DistillRecipe, summary: dict) -> dict:
    overview = dict(summary.get("analysis", {}).get("all_pairs_overview", {}) or {})
    return {
        "recipe": recipe.name,
        "clip_dir": overview.get("clip_dir", float("nan")),
        "clip_style": overview.get("clip_style", float("nan")),
        "clip_content": overview.get("clip_content", float("nan")),
        "content_lpips": overview.get("content_lpips", float("nan")),
        "classifier_acc": overview.get("classifier_acc", float("nan")),
        "num_pairs": 750,
        "avg_inference_seconds": summary.get("avg_inference_seconds", float("nan")),
    }


def run_recipe(
    recipe: DistillRecipe,
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
) -> dict:
    rng = random.Random(seed)
    model, config = _load_checkpoint_model(checkpoint, device)
    model.train()
    for param in model.parameters():
        param.requires_grad_(False)
    model.style_emb.weight.requires_grad_(True)
    model.style_spatial_id_16.requires_grad_(recipe.optimize_spatial)

    latent_index = _style_latent_index(latent_root, style_names)
    content_pool = [p for style in style_names for p in latent_index[style]]
    transport = SWDTransportCost(config)

    params = [model.style_emb.weight]
    if recipe.optimize_spatial:
        params.append(model.style_spatial_id_16)
    optimizer = torch.optim.Adam(params, lr=recipe.lr)

    losses_log: list[dict] = []
    start_time = time.time()
    for style_id in target_style_ids:
        style_name = style_names[style_id]
        for it in range(recipe.calibration_iters):
            optimizer.zero_grad(set_to_none=True)
            content = _sample_latent_batch(content_pool, recipe.batch_size, device, rng)
            target = _sample_latent_batch(latent_index[style_name], recipe.batch_size, device, rng)
            sid = torch.full((recipe.batch_size,), style_id, dtype=torch.long, device=device)
            z = _integrate_with_grad(model, content, style_id=sid, num_steps=recipe.ode_steps)
            delta = z - content
            tv_acc = _tv_loss(delta) if recipe.tv_weight > 0.0 else content.new_tensor(0.0)
            div_acc = content.new_tensor(0.0)
            swd_loss = transport.aligned_cost(z, target)
            grad_loss = _gradient_cosine_loss(z, content) if recipe.grad_weight > 0.0 else content.new_tensor(0.0)
            loss = swd_loss
            if recipe.tv_weight > 0.0:
                loss = loss + recipe.tv_weight * tv_acc
            if recipe.grad_weight > 0.0:
                loss = loss + recipe.grad_weight * grad_loss
            if recipe.div_weight > 0.0:
                loss = loss + recipe.div_weight * div_acc
            if not loss.requires_grad:
                losses_log.append(
                    {
                        "style_id": style_id,
                        "style_name": style_name,
                        "iter": it + 1,
                        "loss": float(loss.detach().item()),
                        "swd": float(swd_loss.detach().item()),
                        "tv": float(tv_acc.detach().item()),
                        "grad": float(grad_loss.detach().item()),
                        "div": float(div_acc.detach().item()),
                        "no_grad": 1,
                    }
                )
                print(
                    f"[{recipe.name}] style={style_name} iter={it + 1}/{recipe.calibration_iters} "
                    "has no trainable gradient path; skipping remaining iterations for this style"
                )
                break
            loss.backward()
            optimizer.step()
            losses_log.append(
                {
                    "style_id": style_id,
                    "style_name": style_name,
                    "iter": it + 1,
                    "loss": float(loss.detach().item()),
                    "swd": float(swd_loss.detach().item()),
                    "tv": float(tv_acc.detach().item()),
                    "grad": float(grad_loss.detach().item()),
                    "div": float(div_acc.detach().item()),
                    "no_grad": 0,
                }
            )
            if (it + 1) % 20 == 0 or it == 0 or (it + 1) == recipe.calibration_iters:
                print(
                    f"[{recipe.name}] style={style_name} iter={it + 1}/{recipe.calibration_iters} "
                    f"loss={loss.detach().item():.4f} swd={swd_loss.detach().item():.4f} "
                    f"tv={tv_acc.detach().item():.4f} "
                    f"grad={grad_loss.detach().item():.4f}"
                )

    recipe_dir = out_root / recipe.name
    adapter_path = recipe_dir / "style_adapter.pt"
    _save_style_adapter(adapter_path, model)
    _write_json(
        recipe_dir / "distill_config.json",
        {
            "recipe": recipe.__dict__,
            "checkpoint": str(checkpoint),
            "latent_root": str(latent_root),
            "style_names": style_names,
            "target_style_ids": target_style_ids,
            "vae_model": _infer_vae_model(config, vae_model),
            "elapsed_seconds": time.time() - start_time,
        },
    )
    with (recipe_dir / "distill_losses.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(losses_log[0].keys()))
        writer.writeheader()
        writer.writerows(losses_log)

    full_eval_dir = recipe_dir / "full_eval"
    summary = _run_full_eval(checkpoint, adapter_path, full_eval_dir, eval_batch_size, vae_model=vae_model)
    _write_json(recipe_dir / "full_eval_summary.json", summary)
    return {
        "recipe": recipe.__dict__,
        "adapter_path": str(adapter_path),
        "full_eval_dir": str(full_eval_dir),
        "summary": summary,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Distill standalone style adapters without modifying the base checkpoint.")
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "exp/diffeomorphic_tangent_sweep/t00_ws0p03_g6_nl0/epoch_0008.pt")
    parser.add_argument("--latent-root", type=Path, default=ROOT.parent / "latent-256")
    parser.add_argument("--out-root", type=Path, default=ROOT / "exp/style_embedding_distill/t00_epoch8")
    parser.add_argument("--style-subdirs", type=str, default="photo,Hayao,monet,vangogh,cezanne")
    parser.add_argument("--target-style-ids", type=str, default="1,2,3,4")
    parser.add_argument("--recipes", type=str, default="")
    parser.add_argument("--eval-batch-size", type=int, default=0)
    parser.add_argument("--vae-model", type=str, default="auto", help="VAE alias/id for eval generation; default reads checkpoint config.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    style_names = [x.strip() for x in args.style_subdirs.split(",") if x.strip()]
    target_style_ids = [int(x.strip()) for x in args.target_style_ids.split(",") if x.strip()]
    recipes = _parse_recipes(args.recipes)
    eval_batch_size = _memory_tier_eval_batch_size(args.device, args.eval_batch_size if args.eval_batch_size > 0 else None)
    args.out_root.mkdir(parents=True, exist_ok=True)

    all_rows = []
    all_payload = []
    for recipe in recipes:
        payload = run_recipe(
            recipe,
            checkpoint=args.checkpoint,
            latent_root=args.latent_root,
            out_root=args.out_root,
            style_names=style_names,
            target_style_ids=target_style_ids,
            eval_batch_size=eval_batch_size,
            vae_model=args.vae_model,
            seed=args.seed,
            device=args.device,
        )
        all_payload.append(payload)
        all_rows.append(_summary_row(recipe, payload["summary"]))

    summary_csv = args.out_root / "full_eval_summary.csv"
    with summary_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
        writer.writeheader()
        writer.writerows(all_rows)
    _write_json(args.out_root / "summary.json", {"results": all_payload, "rows": all_rows})
    print(f"Saved recipe sweep summary to {summary_csv}")


if __name__ == "__main__":
    main()
