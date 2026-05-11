"""
Batch evaluation for all baselines.
Loads models once, evaluates all baseline+style combos.
Computes: LPIPS, CLIP_Style, CLIP_Content (aligned with SchrodingerBridge).
"""
import os
import sys
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image
from pathlib import Path
import pandas as pd
import gc

SCRIPT_DIR = Path(__file__).parent.resolve()
PIPELINE_ROOT = SCRIPT_DIR.parent
REPO_ROOT = PIPELINE_ROOT.parent.parent
STYLE_DATA = REPO_ROOT / "style_data"
OVERFIT50 = STYLE_DATA / "overfit50"
CACHE_DIR = REPO_ROOT / "Cycle-NCE" / "eval_cache"
LOCAL_CLIP_DIR = CACHE_DIR / "manual_clip" / "openai-clip-vit-base-patch32"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32
IMAGE_SIZE = 256
BATCH_SIZE = 4

ALL_STYLES = ["monet", "vangogh", "ukiyoe", "cezanne", "Hayao"]
BASELINE_STYLES = {
    "s2wat": ["monet", "vangogh", "cezanne", "Hayao"],
    "samst": ["monet", "vangogh", "cezanne", "ukiyoe"],
    "styleid": ["monet", "vangogh", "cezanne", "Hayao"],
    "cut": ["monet", "vangogh", "cezanne", "Hayao"],
}

transform = T.Compose([
    T.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    T.ToTensor(),
    T.Normalize([0.5]*3, [0.5]*3)
])


def load_images(dir_path, transform_fn):
    imgs, paths = [], []
    for f in sorted(Path(dir_path).glob("*.jpg")) + sorted(Path(dir_path).glob("*.png")):
        img = Image.open(f).convert("RGB")
        imgs.append(transform_fn(img))
        paths.append(f.name)
    if not imgs:
        return torch.empty(0), []
    return torch.stack(imgs).to(DEVICE, dtype=DTYPE), paths


def compute_lpips(content_imgs, gen_imgs, lpips_model):
    n = len(gen_imgs)
    scores = []
    for i in range(0, n, BATCH_SIZE):
        with torch.no_grad():
            s = lpips_model(content_imgs[i:i+BATCH_SIZE], gen_imgs[i:i+BATCH_SIZE]).squeeze()
        scores.extend(s.cpu().tolist() if s.dim() > 0 else [s.item()])
    return sum(scores) / len(scores)


def compute_clip_feats(clip_model, clip_processor, dir_path):
    imgs = []
    for f in sorted(Path(dir_path).glob("*.jpg")) + sorted(Path(dir_path).glob("*.png")):
        img = Image.open(f).convert("RGB")
        imgs.append(img)
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


def main():
    output = PIPELINE_ROOT / "results" / "metrics_batch.csv"

    # Load models once
    print("Loading LPIPS...")
    import lpips
    lpips_model = lpips.LPIPS(net="alex").to(DEVICE, dtype=DTYPE)
    lpips_model.eval()

    print("Loading CLIP...")
    from transformers import CLIPModel, CLIPProcessor
    clip_src = str(LOCAL_CLIP_DIR) if LOCAL_CLIP_DIR.exists() else "openai/clip-vit-base-patch32"
    clip_model = CLIPModel.from_pretrained(clip_src).to(DEVICE, dtype=DTYPE).eval()
    clip_processor = CLIPProcessor.from_pretrained(clip_src)

    # Precompute content image features (shared across all baselines)
    content_dir = OVERFIT50 / "photo"
    print("Precomputing content CLIP features...")
    content_clip_feats = compute_clip_feats(clip_model, clip_processor, content_dir)

    results = []

    for baseline, styles in BASELINE_STYLES.items():
        for style in styles:
            result_dir = PIPELINE_ROOT / "results" / baseline / style
            style_dir = OVERFIT50 / style

            if not result_dir.exists() or not any(result_dir.glob("*.jpg")):
                print(f"[SKIP] {baseline}/{style} - no images")
                continue

            print(f"\n[EVAL] {baseline}/{style}")
            metrics = {
                "baseline": baseline,
                "style": style,
                "resolution": IMAGE_SIZE,
            }

            # Load images
            content_imgs, _ = load_images(content_dir, transform)
            gen_imgs, gen_paths = load_images(result_dir, transform)

            if len(gen_imgs) == 0:
                print(f"  [SKIP] No generated images")
                continue

            n = min(len(content_imgs), len(gen_imgs))
            content_imgs = content_imgs[:n]
            gen_imgs = gen_imgs[:n]

            # LPIPS
            try:
                lpips_score = compute_lpips(content_imgs, gen_imgs, lpips_model)
                metrics["lpips"] = lpips_score
                print(f"  LPIPS: {lpips_score:.4f}")
            except Exception as e:
                print(f"  [WARN] LPIPS failed: {e}")

            del content_imgs, gen_imgs
            gc.collect(); torch.cuda.empty_cache()

            # CLIP
            try:
                gen_feats = compute_clip_feats(clip_model, clip_processor, result_dir)
                style_feats = compute_clip_feats(clip_model, clip_processor, style_dir)

                if gen_feats is not None and style_feats is not None:
                    sim = (gen_feats @ style_feats.T).mean().item()
                    metrics["clip_style"] = sim
                    print(f"  CLIP_Style: {sim:.4f}")

                if gen_feats is not None and content_clip_feats is not None:
                    sim = (gen_feats @ content_clip_feats.T).mean().item()
                    metrics["clip_content"] = sim
                    print(f"  CLIP_Content: {sim:.4f}")

                if gen_feats is not None: del gen_feats
                if style_feats is not None: del style_feats
                gc.collect(); torch.cuda.empty_cache()
            except Exception as e:
                print(f"  [WARN] CLIP failed: {e}")

            results.append(metrics)

    # Save CSV
    df = pd.DataFrame(results)
    df.to_csv(output, index=False)
    print(f"\nSaved {len(results)} rows to {output}")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
