#!/usr/bin/env python
r"""Standalone evaluation script for the pixel-space 128x128 SFM checkpoint.

Evaluates a pixel128 ``SpatialBridge620`` checkpoint on a test set:
  1. Loads the model directly from checkpoint (NO VAE — operates on 128x128 RGB
     tensors with ``latent_channels=3``).
  2. Generates stylized outputs: 5 styles x 30 content images x 5 target
     styles = 750 transfers (or 600 transfer-only).
  3. Runs the model's 8-step Euler ODE integration with endpoint AdaIN/WCT
     (``integrate_transport`` with ``target_style_latent``).
  4. Upscales 128 -> 512 bilinear for fair CLIP-S / LPIPS comparison against
     the latent-space version.
  5. Computes CLIP-S (HF CLIP ViT-B/32, cosine sim to target-style prototype)
     and LPIPS (AlexNet, vs content at 512x512).
  6. Saves outputs as PNG and writes ``summary.json``.

VRAM budget: <= 7 GB.  batch_size=1 for generation, batch_size=2 for metrics.

Usage (remote Windows / RTX 3060 12 GB)::

    python scripts\eval_pixel128.py ^
        --checkpoint C:\Users\Administrator\exp\pixel128_sfm\pixel128_b2_e8\epoch_0002.pt ^
        --config configs\630_pixel_128.json ^
        --test_dir I:\wikiart_distinct5_samam_512_classview\test ^
        --output C:\Users\Administrator\exp\pixel128_sfm\eval\epoch_0002 ^
        --clip_cache_dir I:\Github\Latent_Style\eval_cache\hf

The script is self-contained: it only imports from ``src/`` (added to
``sys.path`` automatically) and standard pip packages (torch, numpy, PIL,
transformers, lpips).
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

# --- Make ``src/`` importable -------------------------------------------------
_SRC_ROOT = Path(__file__).resolve().parents[1] / "src"
if str(_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(_SRC_ROOT))

from config_schema import load_experiment_config  # noqa: E402
from model import build_model_from_config  # noqa: E402
from style_families import prune_state_dict_for_tokenizer_family  # noqa: E402
from utils.training import strip_compile_prefix  # noqa: E402

# --- Constants ----------------------------------------------------------------
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
PIXEL_SIZE = 128        # model native resolution (default; override via --pixel_size)
EVAL_SIZE = 512         # upscale target for fair CLIP-S / LPIPS
CLIP_REPO = "openai/clip-vit-base-patch32"
CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)
CLIP_IMAGE_SIZE = 224
METRIC_BATCH = 2        # batch_size for CLIP / LPIPS


# --- Image helpers ------------------------------------------------------------
def load_image_tensor(path: Path, size: int = PIXEL_SIZE) -> torch.Tensor:
    """Load image, resize to ``size``x``size`` LANCZOS, return (3, H, W) in [-1, 1]."""
    img = Image.open(path).convert("RGB").resize((size, size), Image.LANCZOS)
    arr = np.asarray(img, dtype=np.uint8)
    t = torch.from_numpy(arr.copy()).permute(2, 0, 1).float()  # (3, H, W)
    t = t / 127.5 - 1.0  # [0,255] -> [-1,1]
    return t.contiguous()


def load_image_01(path: Path, size: int = EVAL_SIZE) -> torch.Tensor:
    """Load image, resize to ``size``x``size`` BILINEAR, return (3, H, W) in [0, 1]."""
    img = Image.open(path).convert("RGB").resize((size, size), Image.BILINEAR)
    arr = np.asarray(img, dtype=np.uint8)
    t = torch.from_numpy(arr.copy()).permute(2, 0, 1).float() / 255.0
    return t.contiguous()


def to_01(t: torch.Tensor) -> torch.Tensor:
    """[-1, 1] -> [0, 1]."""
    return (t + 1.0) / 2.0


def to_lpips_input(t: torch.Tensor) -> torch.Tensor:
    """[0, 1] -> [-1, 1]."""
    return t * 2.0 - 1.0


def save_png(tensor_01: torch.Tensor, path: Path) -> None:
    """Save a (3, H, W) tensor in [0, 1] as PNG."""
    arr = (tensor_01.clamp(0, 1) * 255.0).byte().permute(1, 2, 0).cpu().numpy()
    Image.fromarray(arr).save(str(path))


def prepare_clip_pixels(images_01: torch.Tensor, image_size: int, mean, std) -> torch.Tensor:
    """Center-crop + bicubic resize + normalize for CLIP (replicates run_evaluation)."""
    if images_01.ndim == 3:
        images_01 = images_01.unsqueeze(0)
    imgs = images_01.to(dtype=torch.float32)
    h, w = imgs.shape[-2:]
    crop = min(h, w)
    if h != crop or w != crop:
        top = max(0, (h - crop) // 2)
        left = max(0, (w - crop) // 2)
        imgs = imgs[:, :, top:top + crop, left:left + crop]
    if imgs.shape[-2:] != (image_size, image_size):
        imgs = F.interpolate(
            imgs, size=(image_size, image_size),
            mode="bicubic", align_corners=False, antialias=True,
        )
    mean_t = torch.tensor(mean, device=imgs.device, dtype=imgs.dtype).view(1, 3, 1, 1)
    std_t = torch.tensor(std, device=imgs.device, dtype=imgs.dtype).view(1, 3, 1, 1)
    return ((imgs - mean_t) / std_t).contiguous()


def extract_clip_features(clip_model, pixel_values: torch.Tensor) -> torch.Tensor:
    """Call CLIP get_image_features and extract a plain tensor.

    Some transformers versions return BaseModelOutputWithPooling instead of
    a raw tensor; handle both cases.
    """
    out = clip_model.get_image_features(pixel_values=pixel_values)
    if torch.is_tensor(out):
        return out
    # HuggingFace output object fallback
    if hasattr(out, "pooler_output") and out.pooler_output is not None:
        return out.pooler_output
    if hasattr(out, "last_hidden_state"):
        return out.last_hidden_state[:, 0]
    if hasattr(out, "image_embeds") and out.image_embeds is not None:
        return out.image_embeds
    raise TypeError(f"Unexpected CLIP output type: {type(out)}")


# --- Model loading ------------------------------------------------------------
def build_model_from_checkpoint(
    checkpoint_path: str,
    config_path: str,
    device: torch.device,
) -> tuple[torch.nn.Module, object]:
    """Load config, build model, load checkpoint state_dict (strict=False fallback)."""
    config = load_experiment_config(config_path)
    model = build_model_from_config(
        config.model, bridge_cfg=config.bridge, use_checkpointing=False,
    ).to(device)

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dict = strip_compile_prefix(checkpoint["model_state_dict"])
    state_dict, _removed = prune_state_dict_for_tokenizer_family(
        state_dict,
        tokenizer_family=str(getattr(config.model, "tokenizer_family", "legacy_factorized")),
        contract_family=str(getattr(config.model, "contract_family", "legacy")),
        style_injection_mode=str(getattr(config.model, "style_injection_mode", "none")),
        proximal_mode=str(getattr(config.model, "proximal_mode", "off")),
        style_delta_mode=str(getattr(config.model, "style_delta_mode", "none")),
        output_appearance_alignment_mode=str(getattr(config.model, "output_appearance_alignment_mode", "none")),
    )
    try:
        model.load_state_dict(state_dict, strict=True)
    except RuntimeError:
        model.load_state_dict(state_dict, strict=False)
    model.eval()
    return model, config


# --- Test image discovery -----------------------------------------------------
def discover_test_images(
    test_dir: Path,
    style_subdirs: list[str],
    max_per_style: int = 30,
    seed: int = 42,
) -> dict[int, tuple[str, list[Path]]]:
    """Discover test images per style, shuffled with fixed seed (matches run_evaluation)."""
    result: dict[int, tuple[str, list[Path]]] = {}
    for sid, sname in enumerate(style_subdirs):
        s_dir = test_dir / sname
        if not s_dir.exists():
            print(f"  WARNING: style dir not found: {s_dir}")
            continue
        images = sorted(
            p for p in s_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS
        )
        rng = random.Random(seed)
        rng.shuffle(images)
        if max_per_style > 0:
            images = images[:max_per_style]
        result[sid] = (sname, images)
    return result


# --- Main ---------------------------------------------------------------------
def main() -> None:
    global PIXEL_SIZE
    parser = argparse.ArgumentParser(
        description="Standalone evaluation for pixel-space SFM checkpoint (128 or 256)",
    )
    parser.add_argument("--checkpoint", required=True, help="Path to .pt checkpoint")
    parser.add_argument("--config", required=True, help="Path to config JSON")
    parser.add_argument("--test_dir", required=True, help="Test image dir (per-style subdirs)")
    parser.add_argument("--output", required=True, help="Output dir for PNGs + summary.json")
    parser.add_argument("--num_steps", type=int, default=8, help="ODE integration steps")
    parser.add_argument("--step_size", type=float, default=1.0, help="ODE step size (horizon)")
    parser.add_argument("--max_per_style", type=int, default=30, help="Max content images per style")
    parser.add_argument("--transfer_only", action="store_true", help="Skip identity (src==tgt) pairs")
    parser.add_argument("--clip_cache_dir", default="", help="HF cache dir for CLIP model")
    parser.add_argument("--device", default="cuda", help="torch device")
    parser.add_argument("--pixel_size", type=int, default=PIXEL_SIZE, help="Model native resolution (128 or 256)")
    args = parser.parse_args()
    PIXEL_SIZE = args.pixel_size  # override module constant for all functions

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # === 1. Build model ======================================================
    print(f"[1/5] Loading model from {args.checkpoint}")
    t0 = time.perf_counter()
    model, config = build_model_from_checkpoint(args.checkpoint, args.config, device)
    style_subdirs = list(config.data.style_subdirs)
    num_styles = len(style_subdirs)
    print(f"    Styles ({num_styles}): {style_subdirs}")
    print(f"    Model loaded in {time.perf_counter() - t0:.1f}s")

    # === 2. Discover test images =============================================
    print(f"[2/5] Discovering test images in {args.test_dir}")
    test_images = discover_test_images(
        Path(args.test_dir), style_subdirs, args.max_per_style,
    )
    total_content = sum(len(v[1]) for v in test_images.values())
    print(f"    {total_content} content images across {len(test_images)} styles")

    # === 3. Pick reference latent per target style ===========================
    # Use the first image (after seed-42 shuffle) of each target style as the
    # target_style_latent for the N1 endpoint AdaIN/WCT block.
    ref_latents: dict[int, torch.Tensor] = {}
    for sid, (_sname, imgs) in test_images.items():
        if imgs:
            ref = load_image_tensor(imgs[0], PIXEL_SIZE).unsqueeze(0).to(device)
            ref_latents[sid] = ref

    # === 4. Generation phase (batch all target styles per content) ===========
    pair_mode = "transfer-only" if args.transfer_only else "all-pairs"
    print(f"[3/5] Generating stylized outputs ({pair_mode}) [batched targets]")
    gen_start = time.perf_counter()
    generated: list[dict] = []
    pair_count = 0
    pairs_per_content = num_styles - 1 if args.transfer_only else num_styles
    total_pairs = total_content * pairs_per_content

    for src_sid, (src_sname, src_imgs) in test_images.items():
        for src_path in src_imgs:
            content_128 = load_image_tensor(src_path, PIXEL_SIZE).unsqueeze(0).to(device)
            # Build target list (skip identity if transfer_only)
            tgt_list = [s for s in range(num_styles) if not (args.transfer_only and s == src_sid)]
            n_tgt = len(tgt_list)
            # Expand content to batch=n_tgt, stack style_id and target_style_latent
            content_batch = content_128.expand(n_tgt, -1, -1, -1).contiguous()
            style_ids = torch.tensor(tgt_list, dtype=torch.long, device=device)
            target_refs = torch.stack([ref_latents[s] for s in tgt_list], dim=0).to(device)
            with torch.no_grad():
                gen_batch = model.integrate(
                    content_batch,
                    style_id=style_ids,
                    num_steps=args.num_steps,
                    step_size=args.step_size,
                    target_style_latent=target_refs,
                )
            # Save each output
            for i, tgt_sid in enumerate(tgt_list):
                pair_count += 1
                gen_01 = to_01(gen_batch[i].detach()).clamp(0, 1)
                out_name = f"{src_sname}__{src_path.stem}_to_{style_subdirs[tgt_sid]}.png"
                out_path = output_dir / out_name
                save_png(gen_01, out_path)
                generated.append({
                    "content_path": str(src_path),
                    "src_style_id": int(src_sid),
                    "src_style": src_sname,
                    "tgt_style_id": int(tgt_sid),
                    "tgt_style": style_subdirs[tgt_sid],
                    "gen_path": str(out_path),
                })
            if pair_count % 10 == 0 or pair_count >= total_pairs:
                elapsed = time.perf_counter() - gen_start
                print(
                    f"    [{pair_count}/{total_pairs}] "
                    f"{src_sname} ({elapsed:.1f}s, {pair_count / max(elapsed, 1e-6):.1f} img/s)"
                )

    gen_time = time.perf_counter() - gen_start
    print(f"    Generation done: {pair_count} pairs in {gen_time:.1f}s")

    # Free model VRAM before loading CLIP / LPIPS
    del model
    torch.cuda.empty_cache()

    # === 5. Metrics phase (batch_size=2) =====================================
    print(f"[4/5] Computing CLIP-S + LPIPS at {EVAL_SIZE}x{EVAL_SIZE}")
    metric_start = time.perf_counter()
    clip_scores: list[float] = []
    lpips_scores: list[float] = []

    # --- 5a. Load CLIP -------------------------------------------------------
    clip_model = None
    clip_ok = False
    try:
        from transformers import CLIPModel
        clip_kwargs: dict = {}
        if args.clip_cache_dir:
            clip_kwargs["cache_dir"] = args.clip_cache_dir
        clip_model = CLIPModel.from_pretrained(CLIP_REPO, **clip_kwargs).to(device).eval()
        clip_ok = True
        print(f"    CLIP loaded from {CLIP_REPO}")
    except Exception as exc:
        print(f"    CLIP load FAILED: {exc}")

    # --- 5b. Build CLIP prototypes from reference images ---------------------
    prototypes: dict[int, torch.Tensor] = {}
    if clip_ok and clip_model is not None:
        for sid, (sname, imgs) in test_images.items():
            clips: list[torch.Tensor] = []
            for img_path in imgs:
                img_01 = load_image_01(img_path, EVAL_SIZE)
                pixels = prepare_clip_pixels(
                    img_01, CLIP_IMAGE_SIZE, CLIP_MEAN, CLIP_STD,
                ).to(device)
                with torch.no_grad():
                    feat = extract_clip_features(clip_model, pixels)
                feat = feat.to(dtype=torch.float32)
                if feat.ndim == 1:
                    feat = feat.unsqueeze(0)
                feat = feat / (feat.norm(p=2, dim=-1, keepdim=True) + 1e-8)
                clips.append(feat)
            if clips:
                stacked = torch.cat(clips, dim=0)
                proto = stacked.mean(dim=0, keepdim=True)
                proto = proto / (proto.norm(p=2, dim=-1, keepdim=True) + 1e-8)
                prototypes[sid] = proto.to(device, dtype=torch.float32)
        print(f"    Built prototypes for {len(prototypes)} styles")

    # --- 5c. Load LPIPS ------------------------------------------------------
    lpips_fn = None
    lpips_ok = False
    try:
        import lpips
        lpips_fn = lpips.LPIPS(net="alex", verbose=False).to(device).eval()
        lpips_ok = True
        print("    LPIPS (AlexNet) loaded")
    except Exception as exc:
        print(f"    LPIPS load FAILED: {exc}")

    # --- 5d. Single pass: CLIP-S + LPIPS in batches of METRIC_BATCH ----------
    for i in range(0, len(generated), METRIC_BATCH):
        batch = generated[i:i + METRIC_BATCH]
        # Reload generated PNGs at 128, upscale to 512 bilinear
        gen_01_batch = []
        for g in batch:
            gen_128 = load_image_01(Path(g["gen_path"]), PIXEL_SIZE)
            gen_512 = F.interpolate(
                gen_128.unsqueeze(0), size=(EVAL_SIZE, EVAL_SIZE),
                mode="bilinear", align_corners=False,
            ).squeeze(0)
            gen_01_batch.append(gen_512)
        gen_batch = torch.stack(gen_01_batch, dim=0).to(device)

        # Load content images at 512
        content_batch = torch.stack(
            [load_image_01(Path(g["content_path"]), EVAL_SIZE) for g in batch],
            dim=0,
        ).to(device)

        # --- CLIP-S ---
        if clip_ok and clip_model is not None:
            pixels = prepare_clip_pixels(
                gen_batch, CLIP_IMAGE_SIZE, CLIP_MEAN, CLIP_STD,
            )
            with torch.no_grad():
                feats = extract_clip_features(clip_model, pixels)
            feats = feats.to(dtype=torch.float32)
            if feats.ndim == 1:
                feats = feats.unsqueeze(0)
            feats = feats / (feats.norm(p=2, dim=-1, keepdim=True) + 1e-8)
            for j, g in enumerate(batch):
                proto = prototypes.get(g["tgt_style_id"])
                if proto is not None and proto.shape[-1] == feats.shape[-1]:
                    score = float(
                        F.cosine_similarity(
                            feats[j:j + 1].float(), proto, dim=-1,
                        ).item()
                    )
                else:
                    score = 0.0
                g["clip_style"] = score
                clip_scores.append(score)

        # --- LPIPS ---
        if lpips_ok and lpips_fn is not None:
            gen_lpips = to_lpips_input(gen_batch)
            content_lpips = to_lpips_input(content_batch)
            with torch.no_grad():
                d = lpips_fn(gen_lpips, content_lpips)
            for j, g in enumerate(batch):
                score = float(d[j].item())
                g["content_lpips"] = score
                lpips_scores.append(score)

        done = min(i + METRIC_BATCH, len(generated))
        if done % 10 == 0 or done == len(generated):
            elapsed = time.perf_counter() - metric_start
            print(f"    [{done}/{len(generated)}] metrics ({elapsed:.1f}s)")

    metric_time = time.perf_counter() - metric_start
    print(f"    Metrics done in {metric_time:.1f}s")

    # Free metric models
    if clip_model is not None:
        del clip_model
    if lpips_fn is not None:
        del lpips_fn
    torch.cuda.empty_cache()

    # === 6. Summary ==========================================================
    print("[5/5] Writing summary")
    total_time = gen_time + metric_time
    summary = {
        "checkpoint": str(args.checkpoint),
        "config": str(args.config),
        "test_dir": str(args.test_dir),
        "num_steps": int(args.num_steps),
        "step_size": float(args.step_size),
        "transfer_only": bool(args.transfer_only),
        "num_pairs": len(generated),
        "pixel_size": PIXEL_SIZE,
        "eval_size": EVAL_SIZE,
        "clip_style_mean": float(np.mean(clip_scores)) if clip_scores else None,
        "clip_style_std": float(np.std(clip_scores)) if clip_scores else None,
        "content_lpips_mean": float(np.mean(lpips_scores)) if lpips_scores else None,
        "content_lpips_std": float(np.std(lpips_scores)) if lpips_scores else None,
        "timing": {
            "generation_s": round(gen_time, 2),
            "metrics_s": round(metric_time, 2),
            "total_s": round(total_time, 2),
        },
    }

    # Per-target-style breakdown
    per_style: dict[str, dict[str, list[float]]] = {}
    for g in generated:
        key = g["tgt_style"]
        per_style.setdefault(key, {"clip_style": [], "content_lpips": []})
        if "clip_style" in g:
            per_style[key]["clip_style"].append(g["clip_style"])
        if "content_lpips" in g:
            per_style[key]["content_lpips"].append(g["content_lpips"])
    summary["per_target_style"] = {
        k: {
            "clip_style_mean": float(np.mean(v["clip_style"])) if v["clip_style"] else None,
            "content_lpips_mean": float(np.mean(v["content_lpips"])) if v["content_lpips"] else None,
            "count": len(v["clip_style"]),
        }
        for k, v in per_style.items()
    }

    summary_path = output_dir / "summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    # Per-pair CSV
    csv_path = output_dir / "metrics.csv"
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        f.write("src_style,tgt_style,content_path,gen_path,clip_style,content_lpips\n")
        for g in generated:
            f.write(
                f"{g['src_style']},{g['tgt_style']},"
                f"\"{g['content_path']}\",\"{g['gen_path']}\","
                f"{g.get('clip_style', '')},{g.get('content_lpips', '')}\n"
            )

    # Console summary
    print(f"\n{'=' * 60}")
    print(f"PIXEL128 SFM EVALUATION SUMMARY")
    print(f"{'=' * 60}")
    print(f"Checkpoint:      {args.checkpoint}")
    print(f"Pairs:           {len(generated)} ({pair_mode})")
    if clip_scores:
        print(f"CLIP-S:          mean={summary['clip_style_mean']:.4f}  std={summary['clip_style_std']:.4f}")
    else:
        print(f"CLIP-S:          N/A")
    if lpips_scores:
        print(f"LPIPS:           mean={summary['content_lpips_mean']:.4f}  std={summary['content_lpips_std']:.4f}")
    else:
        print(f"LPIPS:           N/A")
    print(f"Generation:      {gen_time:.1f}s")
    print(f"Metrics:         {metric_time:.1f}s")
    print(f"Total:           {total_time:.1f}s")
    print(f"{'=' * 60}")
    print(f"Per-target-style:")
    for sname in style_subdirs:
        ps = summary["per_target_style"].get(sname, {})
        cs = ps.get("clip_style_mean")
        lp = ps.get("content_lpips_mean")
        cs_str = f"{cs:.4f}" if cs is not None else "N/A"
        lp_str = f"{lp:.4f}" if lp is not None else "N/A"
        print(f"  {sname:20s}  CLIP-S={cs_str}  LPIPS={lp_str}  ({ps.get('count', 0)} pairs)")
    print(f"{'=' * 60}")
    print(f"Summary:  {summary_path}")
    print(f"CSV:      {csv_path}")


if __name__ == "__main__":
    main()
