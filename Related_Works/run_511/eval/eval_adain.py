"""Quick evaluation for AdaIN 750-image outputs.

Computes CLIP-style, CLIP-content, LPIPS-content matching the
SchrodingerBridge evaluation protocol.
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
WORKSPACE_ROOT = THIS_DIR.parents[2]
OVERFIT50 = WORKSPACE_ROOT / "style_data" / "overfit50"
STYLES = ["photo", "monet", "vangogh", "cezanne", "Hayao"]
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32
LPIPS_DTYPE = torch.float32  # LPIPS produces NaN with float16
IMG_SIZE = 256


def load_img(path: Path) -> torch.Tensor:
    transform = T.Compose([T.Resize((IMG_SIZE, IMG_SIZE)), T.ToTensor(), T.Normalize([0.5]*3, [0.5]*3)])
    return transform(Image.open(path).convert("RGB")).unsqueeze(0)


def eval_750(images_dir: Path) -> dict:
    print(f"Evaluating: {images_dir}", flush=True)
    device = torch.device(DEVICE)

    # --- Load models ---
    print("  Loading LPIPS...", flush=True)
    import lpips
    lpips_model = lpips.LPIPS(net="alex").to(device, dtype=LPIPS_DTYPE).eval()

    print("  Loading CLIP...", flush=True)
    from transformers import CLIPModel, CLIPProcessor
    cache_dir = WORKSPACE_ROOT / "Cycle-NCE" / "eval_cache" / "manual_clip" / "openai-clip-vit-base-patch32"
    clip_src = str(cache_dir) if cache_dir.exists() else "openai/clip-vit-base-patch32"
    clip_model = CLIPModel.from_pretrained(clip_src).to(device, dtype=DTYPE).eval()
    clip_processor = CLIPProcessor.from_pretrained(clip_src)

    def get_clip_feat(out):
        """Extract 1D CLIP feature from model output (handles version differences)."""
        if isinstance(out, torch.Tensor):
            return out
        if hasattr(out, 'pooler_output') and out.pooler_output is not None:
            return out.pooler_output
        return out.last_hidden_state[:, 0, :]

    # --- Collect images by target style ---
    all_images = sorted(images_dir.glob("*.jpg"))
    by_target: dict[str, list[Path]] = {s: [] for s in STYLES}
    for img in all_images:
        stem = img.stem  # e.g. "photo_00001_to_monet"
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

        # Load style reference (first image in overfit50/<target>)
        style_dir = OVERFIT50 / target
        style_imgs = sorted(style_dir.glob("*.jpg"))
        if not style_imgs:
            continue
        style_img = load_img(style_imgs[0]).to(device, dtype=DTYPE)

        # Compute style CLIP features
        style_pil = Image.open(style_imgs[0]).convert("RGB")
        style_clip_in = clip_processor(images=style_pil, return_tensors="pt")
        style_clip_in = {k: v.to(device, dtype=DTYPE) if v.is_floating_point() else v.to(device) for k, v in style_clip_in.items()}
        with torch.no_grad():
            style_feat = F.normalize(get_clip_feat(clip_model.get_image_features(**style_clip_in)).float(), dim=-1)

        target_lpips, target_clip_s, target_clip_c = [], [], []

        for gen_path in gen_paths:
            # Parse content source from filename: "{src_style}_{stem}_to_{target}.jpg"
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

            # LPIPS
            with torch.no_grad():
                lp = lpips_model(content_img, gen_img).item()
            target_lpips.append(lp)

            # CLIP-style
            gen_pil = Image.open(gen_path).convert("RGB")
            gen_clip_in = clip_processor(images=gen_pil, return_tensors="pt")
            gen_clip_in = {k: v.to(device, dtype=DTYPE) if v.is_floating_point() else v.to(device) for k, v in gen_clip_in.items()}
            with torch.no_grad():
                gen_feat = F.normalize(get_clip_feat(clip_model.get_image_features(**gen_clip_in)).float(), dim=-1)
            clip_s = (gen_feat @ style_feat.T).item()
            target_clip_s.append(clip_s)

            # CLIP-content
            content_pil = Image.open(content_path).convert("RGB")
            c_clip_in = clip_processor(images=content_pil, return_tensors="pt")
            c_clip_in = {k: v.to(device, dtype=DTYPE) if v.is_floating_point() else v.to(device) for k, v in c_clip_in.items()}
            with torch.no_grad():
                c_feat = F.normalize(get_clip_feat(clip_model.get_image_features(**c_clip_in)).float(), dim=-1)
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

    # Overall
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
    args = parser.parse_args()

    result = eval_750(args.images_dir.resolve())

    output = args.output or args.images_dir.parent / "eval.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"\nSaved: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
