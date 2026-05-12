"""Universal 750-image evaluation for any run_511 baseline.

Computes CLIP-style, CLIP-content, LPIPS-content matching the current
SchrodingerBridge evaluation protocol:
  - CLIP-style uses the normalized mean prototype of all target-style refs
    (up to --max_ref_cache), not a single style image.
  - LPIPS uses VGG, matching SchrodingerBridge/src/utils/run_evaluation.py.

Usage:
  python run_511/eval_750.py --images_dir run_511/outputs/adain_750/infer_750/images
  python run_511/eval_750.py --images_dir run_511/outputs/styleid_750/infer_750/images
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image


THIS_DIR = Path(__file__).resolve().parent
WORKSPACE_ROOT = THIS_DIR.parent
OVERFIT50 = WORKSPACE_ROOT / "style_data" / "overfit50"
STYLES = ["photo", "monet", "vangogh", "cezanne", "Hayao"]
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32
LPIPS_DTYPE = torch.float32  # LPIPS produces NaN with float16
IMG_SIZE = 256


def load_img(path: Path) -> torch.Tensor:
    transform = T.Compose([T.Resize((IMG_SIZE, IMG_SIZE)), T.ToTensor(), T.Normalize([0.5]*3, [0.5]*3)])
    return transform(Image.open(path).convert("RGB")).unsqueeze(0)


def get_clip_feat(out):
    """Extract 1D CLIP feature from model output (handles version differences)."""
    if isinstance(out, torch.Tensor):
        return out
    if hasattr(out, 'pooler_output') and out.pooler_output is not None:
        return out.pooler_output
    return out.last_hidden_state[:, 0, :]


def encode_clip_pils(clip_model, clip_processor, pils: list[Image.Image], device: torch.device) -> torch.Tensor:
    clip_in = clip_processor(images=pils, return_tensors="pt")
    clip_in = {k: v.to(device) for k, v in clip_in.items()}
    with torch.no_grad():
        feat = get_clip_feat(clip_model.get_image_features(**clip_in)).float()
    return F.normalize(feat, dim=-1)


def build_style_prototypes(clip_model, clip_processor, device: torch.device, max_ref_cache: int) -> dict[str, torch.Tensor]:
    prototypes = {}
    for target in STYLES:
        style_paths = sorted((OVERFIT50 / target).glob("*.jpg"))
        if max_ref_cache > 0:
            style_paths = style_paths[:max_ref_cache]
        if not style_paths:
            continue
        feats = []
        for start in range(0, len(style_paths), 64):
            batch_paths = style_paths[start:start + 64]
            pils = [Image.open(p).convert("RGB") for p in batch_paths]
            feats.append(encode_clip_pils(clip_model, clip_processor, pils, device).detach())
        stacked = torch.cat(feats, dim=0)
        proto = stacked.mean(dim=0, keepdim=True)
        prototypes[target] = F.normalize(proto, dim=-1)
    return prototypes


def eval_750(images_dir: Path, max_ref_cache: int = 256) -> dict:
    print(f"Evaluating: {images_dir}", flush=True)
    device = torch.device(DEVICE)

    # --- Load models ---
    print("  Loading LPIPS...", flush=True)
    import lpips
    lpips_model = lpips.LPIPS(net="vgg").to(device, dtype=LPIPS_DTYPE).eval()

    print("  Loading CLIP...", flush=True)
    from transformers import CLIPModel, CLIPProcessor
    cache_dir = WORKSPACE_ROOT / "Cycle-NCE" / "eval_cache" / "manual_clip" / "openai-clip-vit-base-patch32"
    clip_src = str(cache_dir) if cache_dir.exists() else "openai/clip-vit-base-patch32"
    clip_model = CLIPModel.from_pretrained(clip_src).to(device, dtype=DTYPE).eval()
    clip_processor = CLIPProcessor.from_pretrained(clip_src)
    style_prototypes = build_style_prototypes(clip_model, clip_processor, device, max_ref_cache)

    # --- Collect images by target style ---
    all_images = sorted(images_dir.glob("*.jpg"))
    by_target: dict[str, list[Path]] = {s: [] for s in STYLES}
    for img in all_images:
        stem = img.stem
        if "_to_" not in stem:
            continue
        target = stem.split("_to_")[-1]
        if target in by_target:
            by_target[target].append(img)

    print(f"  Found {len(all_images)} images, per target: {', '.join(f'{s}={len(by_target[s])}' for s in STYLES)}", flush=True)

    results = []
    total_lpips, total_clip_s, total_clip_c = [], [], []

    for target in STYLES:
        gen_paths = sorted(by_target[target])
        if not gen_paths:
            continue

        style_feat = style_prototypes.get(target)
        if style_feat is None:
            continue

        target_lpips, target_clip_s, target_clip_c = [], [], []

        for gen_path in gen_paths:
            fname = gen_path.stem
            prefix = fname[: -len(f"_to_{target}")]
            parts = prefix.split("_", 1)
            if len(parts) < 2:
                continue
            src_style, content_stem = parts[0], parts[1]
            content_path = OVERFIT50 / src_style / f"{content_stem}.jpg"
            if not content_path.exists():
                continue

            gen_img = load_img(gen_path).to(device, dtype=LPIPS_DTYPE)
            content_img = load_img(content_path).to(device, dtype=LPIPS_DTYPE)

            with torch.no_grad():
                lp = lpips_model(content_img, gen_img).item()
            target_lpips.append(lp)

            gen_pil = Image.open(gen_path).convert("RGB")
            gen_feat = encode_clip_pils(clip_model, clip_processor, [gen_pil], device)
            clip_s = (gen_feat @ style_feat.T).item()
            target_clip_s.append(clip_s)

            content_pil = Image.open(content_path).convert("RGB")
            c_feat = encode_clip_pils(clip_model, clip_processor, [content_pil], device)
            clip_c = (gen_feat @ c_feat.T).item()
            target_clip_c.append(clip_c)

        n = len(target_lpips)
        row = {
            "target": target,
            "images": n,
            "lpips": round(sum(target_lpips) / n, 4) if n else 0,
            "clip_style": round(sum(target_clip_s) / n, 4) if n else 0,
            "clip_content": round(sum(target_clip_c) / n, 4) if n else 0,
        }
        results.append(row)
        total_lpips.extend(target_lpips)
        total_clip_s.extend(target_clip_s)
        total_clip_c.extend(target_clip_c)
        print(f"  {target}: n={n}  LPIPS={row['lpips']:.4f}  CLIP-s={row['clip_style']:.4f}  CLIP-c={row['clip_content']:.4f}", flush=True)

    n = len(total_lpips)
    overall = {
        "target": "ALL",
        "images": n,
        "lpips": round(sum(total_lpips) / n, 4) if n else 0,
        "clip_style": round(sum(total_clip_s) / n, 4) if n else 0,
        "clip_content": round(sum(total_clip_c) / n, 4) if n else 0,
    }
    results.append(overall)
    print(f"\n  OVERALL: n={n}  LPIPS={overall['lpips']:.4f}  CLIP-s={overall['clip_style']:.4f}  CLIP-c={overall['clip_content']:.4f}", flush=True)

    return {"results": results}


def main() -> int:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--images_dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--max_ref_cache", type=int, default=256)
    args = parser.parse_args()

    result = eval_750(args.images_dir.resolve(), max_ref_cache=args.max_ref_cache)

    output = args.output or args.images_dir.parent / "eval.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"\nSaved: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
