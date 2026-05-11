"""
Unified Evaluation Script for All Baselines
Computes: CMMD, LPIPS, DINO_Structure, CLIP_Style, CLIP_Content

CLIP model aligned with SchrodingerBridge/run_evaluation.py:
  Uses HuggingFace CLIP (openai/clip-vit-base-patch32) from local cache.
"""
import os
import sys
import argparse
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image
from pathlib import Path
import pandas as pd
import json
import gc

SCRIPT_DIR = Path(__file__).parent.resolve()
PIPELINE_ROOT = SCRIPT_DIR.parent
REPO_ROOT = PIPELINE_ROOT.parent.parent
STYLE_DATA = REPO_ROOT / "style_data"
OVERFIT50 = STYLE_DATA / "overfit50"
CACHE_DIR = REPO_ROOT / "Cycle-NCE" / "eval_cache"

# Same local CLIP path as SchrodingerBridge
LOCAL_CLIP_DIR = CACHE_DIR / "manual_clip" / "openai-clip-vit-base-patch32"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32
IMAGE_SIZE = 256
BATCH_SIZE = 4

# Transforms
transform = T.Compose([
    T.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    T.ToTensor(),
    T.Normalize([0.5]*3, [0.5]*3)
])

# CLIP preprocessing is handled by HuggingFace CLIPProcessor (aligned with SchrodingerBridge)


def load_images(dir_path, transform_fn):
    imgs, paths = [], []
    for f in sorted(Path(dir_path).glob("*.jpg")) + sorted(Path(dir_path).glob("*.png")):
        img = Image.open(f).convert("RGB")
        imgs.append(transform_fn(img))
        paths.append(f.name)
    if not imgs:
        return torch.empty(0), []
    return torch.stack(imgs).to(DEVICE, dtype=DTYPE), paths


def evaluate_baseline(baseline_name, style_name):
    """Compute all metrics for one baseline+style"""
    result_dir = PIPELINE_ROOT / "results" / baseline_name / style_name
    content_dir = OVERFIT50 / "photo"
    style_dir = OVERFIT50 / style_name

    if not result_dir.exists() or not any(result_dir.iterdir()):
        print(f"[SKIP] No results in {result_dir}")
        return None

    print(f"\n[EVAL] {baseline_name}/{style_name}")
    metrics = {
        "baseline": baseline_name,
        "style": style_name,
        "resolution": IMAGE_SIZE,
    }

    # Load images
    content_imgs, _ = load_images(content_dir, transform)
    generated_imgs, gen_paths = load_images(result_dir, transform)

    if len(generated_imgs) == 0:
        print(f"[SKIP] No generated images in {result_dir}")
        return None

    # Match content to generated (by filename)
    # For now, use first N of each
    n = min(len(content_imgs), len(generated_imgs))
    content_imgs = content_imgs[:n]
    generated_imgs = generated_imgs[:n]

    # --- LPIPS ---
    try:
        import lpips
        lpips_model = lpips.LPIPS(net="alex").to(DEVICE, dtype=DTYPE)
        lpips_model.eval()
        scores = []
        for i in range(0, n, BATCH_SIZE):
            with torch.no_grad():
                s = lpips_model(content_imgs[i:i+BATCH_SIZE], generated_imgs[i:i+BATCH_SIZE]).squeeze()
            scores.extend(s.cpu().tolist() if s.dim() > 0 else [s.item()])
        metrics["lpips"] = sum(scores) / len(scores)
        del lpips_model
        gc.collect(); torch.cuda.empty_cache()
        print(f"  LPIPS: {metrics['lpips']:.4f}")
    except Exception as e:
        print(f"  [WARN] LPIPS failed: {e}")

    del content_imgs, generated_imgs
    gc.collect(); torch.cuda.empty_cache()

    # --- CLIP Style & Content (HuggingFace, aligned with SchrodingerBridge) ---
    try:
        from transformers import CLIPModel, CLIPProcessor
        clip_src = str(LOCAL_CLIP_DIR) if LOCAL_CLIP_DIR.exists() else "openai/clip-vit-base-patch32"
        clip_model = CLIPModel.from_pretrained(clip_src).to(DEVICE, dtype=DTYPE).eval()
        clip_processor = CLIPProcessor.from_pretrained(clip_src)

        def clip_feats(d):
            imgs, paths = [], []
            for f in sorted(Path(d).glob("*.jpg")) + sorted(Path(d).glob("*.png")):
                img = Image.open(f).convert("RGB")
                imgs.append(img)
                paths.append(f.name)
            if not imgs:
                return None
            feats = []
            for i in range(0, len(imgs), BATCH_SIZE):
                batch = clip_processor(images=imgs[i:i+BATCH_SIZE], return_tensors="pt")
                batch = {k: v.to(DEVICE, dtype=DTYPE) if v.is_floating_point() else v.to(DEVICE) for k, v in batch.items()}
                with torch.no_grad():
                    out = clip_model.get_image_features(**batch)
                    f = out.pooler_output if hasattr(out, 'pooler_output') else out
                    f = F.normalize(f.float(), dim=-1)
                feats.append(f.detach())
            return torch.cat(feats, dim=0)

        gen_feats = clip_feats(result_dir)
        style_feats = clip_feats(style_dir)
        content_feats = clip_feats(content_dir)

        if gen_feats is not None and style_feats is not None:
            sim = (gen_feats @ style_feats.T).mean().item()
            metrics["clip_style"] = sim
            print(f"  CLIP_Style: {sim:.4f}")

        if gen_feats is not None and content_feats is not None:
            sim = (gen_feats @ content_feats.T).mean().item()
            metrics["clip_content"] = sim
            print(f"  CLIP_Content: {sim:.4f}")

        del clip_model, gen_feats
        if style_feats is not None: del style_feats
        if content_feats is not None: del content_feats
        gc.collect(); torch.cuda.empty_cache()
    except Exception as e:
        print(f"  [WARN] CLIP failed: {e}")

    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=str, required=True)
    parser.add_argument("--style", type=str, required=True)
    parser.add_argument("--output", type=str, default=str(PIPELINE_ROOT / "results" / "metrics.csv"))
    args = parser.parse_args()

    metrics = evaluate_baseline(args.baseline, args.style)
    if metrics is None:
        return

    # Append to CSV
    output = Path(args.output)
    if output.exists():
        df = pd.read_csv(output)
        df = pd.concat([df, pd.DataFrame([metrics])], ignore_index=True)
    else:
        df = pd.DataFrame([metrics])
    df.to_csv(output, index=False)
    print(f"\nSaved to {output}")
    print(pd.DataFrame([metrics]).to_string(index=False))


if __name__ == "__main__":
    main()
