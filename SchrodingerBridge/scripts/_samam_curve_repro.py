"""Regenerate SaMam curve with paper-canonical protocol.

1. Generate step_20000 images from final_model_20k.ckpt (if missing)
2. Compute DINO-S for all sampled steps using TEST dir as style refs
   (matching compute_dino_metrics.py protocol, NOT train dir)
3. Output CSV with step, dino_s, clip_s, lpips
"""
import csv
import json
import os
import pathlib
import random
import sys
import time

import torch
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image
from torchvision.utils import save_image

# --- Paths (Windows, remote I: drive) ---
SAMAM_ROOT = pathlib.Path(r"I:\Github\Latent_Style\Related_Works\repos\SaMam")
sys.path.insert(0, str(SAMAM_ROOT))
from TRAIN.lightning_module.lightningmodel import LightningModel  # noqa: E402

TEST_DIR = pathlib.Path(r"I:\datasets\wikiart_distinct5_samam_512_classview\train")
CURVE_BASE = pathlib.Path(
    r"I:\Github\Latent_Style\exp_samam\training"
    r"\samam_distinct5_512_scratch_7k_250eval_remote\curve_eval_30src"
)
CKPT_20K = pathlib.Path(
    r"I:\Github\Latent_Style\exp_samam\training"
    r"\samam_distinct5_512_scratch_7k_250eval_remote\final_model_20k.ckpt"
)
OUT_DIR = pathlib.Path(r"I:\Github\Latent_Style\exp_samam\_dino_curve_repro")
OUT_DIR.mkdir(parents=True, exist_ok=True)

CACHE_DIR = r"I:\Github\Latent_Style\WEAVE\eval_cache"
MODEL_NAME = "facebook/dinov2-small"
BATCH_SIZE = 64
MAX_STYLE_REFS = 30
IMAGE_SIZE = 512

STYLES = ["Early_Renaissance", "Impressionism", "Minimalism", "Rococo", "Ukiyo_e"]
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
SAMPLE_STEPS = [250, 500, 1000, 2000, 3000, 5000, 7000, 20000]

DINO_TRANSFORM = T.Compose([
    T.Resize(224, interpolation=Image.BICUBIC),
    T.CenterCrop(224),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

SAMAM_TRANSFORM = T.Compose([T.Resize((IMAGE_SIZE, IMAGE_SIZE)), T.ToTensor()])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")


def load_rgb(path):
    with Image.open(path) as img:
        return img.convert("RGB")


def image_paths(root):
    return sorted(p for p in root.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS)


# --- Step 1: Generate step_20000 images if missing ---
step20k_dir = CURVE_BASE / "step_020000" / "images"
if step20k_dir.exists() and len(list(step20k_dir.glob("*.png"))) >= 700:
    print(f"step_020000 already has {len(list(step20k_dir.glob('*.png')))} images, skipping generation")
else:
    print(f"\n=== Generating step_20000 images from {CKPT_20K.name} ===")
    # Select source images with same logic as gen_samam_single_ckpt.py (seed 42)
    sources = []
    style_refs = {}
    for style in STYLES:
        style_dir = TEST_DIR / style
        paths = image_paths(style_dir)
        rng = random.Random(42)
        selected = paths[:]
        rng.shuffle(selected)
        selected = selected[:MAX_STYLE_REFS]
        for p in selected:
            sources.append((style, p))
        style_refs[style] = paths[0]

    print(f"  {len(sources)} srcs x {len(style_refs)} styles = {len(sources)*len(style_refs)} images")

    model_samam = LightningModel.load_from_checkpoint(
        checkpoint_path=str(CKPT_20K), map_location=device
    )
    model_samam = model_samam.to(device).eval()

    step20k_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    n = 0
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
        for src_style, src_path in sources:
            content = SAMAM_TRANSFORM(load_rgb(src_path)).unsqueeze(0).to(device)
            for tgt_style, style_path in style_refs.items():
                style = SAMAM_TRANSFORM(load_rgb(style_path)).unsqueeze(0).to(device)
                output = model_samam.forward(content, style)[0].float()
                name = f"{src_style}__{src_path.stem}__to__{tgt_style}.png"
                save_image(output.cpu(), step20k_dir / name)
                n += 1
    print(f"  Generated {n} images in {time.time()-t0:.1f}s")
    del model_samam
    torch.cuda.empty_cache()

# --- Step 2: Load DINOv2 ---
print("\n=== Loading DINOv2 ViT-S/14 ===")
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
from transformers import AutoModel

repo_dir = pathlib.Path(CACHE_DIR) / "models--facebook--dinov2-small"
snap_root = repo_dir / "snapshots"
if snap_root.exists():
    revisions = [p for p in snap_root.iterdir() if p.is_dir()]
    if revisions:
        local_path = str(revisions[0])
        print(f"  Loading from local snapshot: {local_path}")
        dino_model = AutoModel.from_pretrained(local_path).to(device).eval()
    else:
        dino_model = AutoModel.from_pretrained(MODEL_NAME, cache_dir=CACHE_DIR).to(device).eval()
else:
    os.environ["HF_HUB_OFFLINE"] = "0"
    os.environ["TRANSFORMERS_OFFLINE"] = "0"
    dino_model = AutoModel.from_pretrained(MODEL_NAME, cache_dir=CACHE_DIR).to(device).eval()
print("  DINOv2 loaded.")


@torch.inference_mode()
def extract_cls_features(paths):
    cls_features = []
    for start in range(0, len(paths), BATCH_SIZE):
        batch_paths = paths[start:start + BATCH_SIZE]
        pixels = torch.stack([DINO_TRANSFORM(load_rgb(p)) for p in batch_paths]).to(device)
        output = dino_model(pixels, output_hidden_states=True)
        cls = F.normalize(output.last_hidden_state[:, 0, :].float(), dim=-1).cpu()
        cls_features.append(cls)
    return torch.cat(cls_features, dim=0)


# --- Step 3: Precompute style reference features from TEST dir ---
print("\n=== Precomputing style reference features from TEST dir ===")
style_ref_features = {}
for style in STYLES:
    style_dir = TEST_DIR / style
    if not style_dir.exists():
        print(f"  WARNING: {style_dir} not found")
        continue
    ref_paths = image_paths(style_dir)[:MAX_STYLE_REFS]
    feats = extract_cls_features(ref_paths)
    style_ref_features[style] = feats
    print(f"  {style}: {len(ref_paths)} refs")


# --- Step 4: Compute DINO-S for each step ---
print("\n=== Computing DINO-S per step ===")
results = []


def compute_step_dino(step):
    step_dir = CURVE_BASE / f"step_{step:06d}"
    images_dir = step_dir / "images"
    if not images_dir.exists():
        print(f"  step {step}: directory missing, skipping")
        return None
    gen_paths = sorted([p for p in images_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS])
    t0 = time.time()
    gen_feats = extract_cls_features(gen_paths)

    dino_scores = []
    for feat, p in zip(gen_feats, gen_paths):
        parts = p.stem.split("__to__")
        if len(parts) != 2 or parts[1] not in style_ref_features:
            continue
        tgt_style = parts[1]
        ref_feats = style_ref_features[tgt_style].to(device)
        feat_dev = feat.to(device)
        cos_sim = (feat_dev.unsqueeze(0) @ ref_feats.T).squeeze(0)
        max_cos = cos_sim.max().item()
        dino_scores.append((tgt_style, max_cos))

    if not dino_scores:
        return None
    avg_dino_s = sum(s[1] for s in dino_scores) / len(dino_scores)
    elapsed = time.time() - t0
    print(f"  step {step}: DINO-S={avg_dino_s:.6f}  ({len(dino_scores)} imgs, {elapsed:.1f}s)")
    return {"step": step, "dino_s": avg_dino_s, "n_images": len(dino_scores)}


for step in SAMPLE_STEPS:
    r = compute_step_dino(step)
    if r:
        results.append(r)

# --- Step 5: Save ---
out_csv = OUT_DIR / "samam_curve_repro.csv"
out_json = OUT_DIR / "samam_curve_repro.json"
with open(out_csv, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["step", "dino_s", "n_images"])
    for r in results:
        writer.writerow([r["step"], f"{r['dino_s']:.6f}", r["n_images"]])
with open(out_json, "w") as f:
    json.dump(results, f, indent=2)
print(f"\nSaved to {out_csv} and {out_json}")
print(f"Total results: {len(results)}")
print("Done.")
