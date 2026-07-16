"""Evaluate SaMST curve CLIP-S and LPIPS using test dir as style refs.

For each epoch (5, 10, 15), compute:
- CLIP-S: cosine sim between generated image and nearest test-dir style ref (CLIP image features)
- LPIPS: perceptual distance between generated image and source content image
"""
import csv
import json
import os
import pathlib
import sys
import time

import torch
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image

TEST_DIR = pathlib.Path(r"I:\datasets\wikiart_distinct5_512_images\test")
TRAIN_DIR = pathlib.Path(r"I:\datasets\wikiart_distinct5_samam_512_classview\train")
IMG_DIR = pathlib.Path(r"I:\Github\Latent_Style\exp_samam\samst_curve_imgs")
OUT_DIR = pathlib.Path(r"I:\Github\Latent_Style\exp_samam\_dino_curve_repro")
OUT_DIR.mkdir(parents=True, exist_ok=True)

CACHE_DIR = r"I:\Github\Latent_Style\WEAVE\eval_cache"
CLIP_MODEL_NAME = "openai/clip-vit-base-patch16"
BATCH_SIZE = 32

STYLES = ["Early_Renaissance", "Impressionism", "Minimalism", "Rococo", "Ukiyo_e"]
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
EPOCHS = [5, 10, 15]

# CLIP preprocessing (standard CLIP normalize)
CLIP_TRANSFORM = T.Compose([
    T.Resize(224, interpolation=Image.BICUBIC),
    T.CenterCrop(224),
    T.ToTensor(),
    T.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711]),
])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")


def load_rgb(path):
    with Image.open(path) as img:
        return img.convert("RGB")


def image_paths(root):
    return sorted(p for p in root.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS)


# --- Load CLIP model ---
print("=== Loading CLIP ViT-B/16 ===")
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
from transformers import CLIPModel, CLIPProcessor

repo_dir = pathlib.Path(CACHE_DIR) / "models--openai--clip-vit-base-patch16"
snap_root = repo_dir / "snapshots"
if snap_root.exists():
    revisions = [p for p in snap_root.iterdir() if p.is_dir()]
    if revisions:
        local_path = str(revisions[0])
        print(f"  Loading CLIP from local snapshot: {local_path}")
        clip_model = CLIPModel.from_pretrained(local_path).to(device).eval()
    else:
        clip_model = CLIPModel.from_pretrained(CLIP_MODEL_NAME, cache_dir=CACHE_DIR).to(device).eval()
else:
    os.environ["HF_HUB_OFFLINE"] = "0"
    os.environ["TRANSFORMERS_OFFLINE"] = "0"
    clip_model = CLIPModel.from_pretrained(CLIP_MODEL_NAME, cache_dir=CACHE_DIR).to(device).eval()
print("  CLIP loaded.")


# --- Load LPIPS ---
print("=== Loading LPIPS ===")
import lpips
lpips_fn = lpips.LPIPS(net='alex').to(device).eval()
print("  LPIPS loaded.")


@torch.inference_mode()
def extract_clip_image_features(paths):
    feats = []
    for start in range(0, len(paths), BATCH_SIZE):
        batch_paths = paths[start:start + BATCH_SIZE]
        pixels = torch.stack([CLIP_TRANSFORM(load_rgb(p)) for p in batch_paths]).to(device)
        feat = clip_model.get_image_features(pixels)
        feat = F.normalize(feat.float(), dim=-1).cpu()
        feats.append(feat)
    return torch.cat(feats, dim=0)


@torch.inference_mode()
def compute_lpips_batch(gen_paths, src_paths):
    """Compute LPIPS between generated images and their source images."""
    results = []
    for start in range(0, len(gen_paths), BATCH_SIZE):
        batch_gen = gen_paths[start:start + BATCH_SIZE]
        batch_src = src_paths[start:start + BATCH_SIZE]
        gen_tensors = []
        src_tensors = []
        for gp, sp in zip(batch_gen, batch_src):
            gen_img = load_rgb(gp).resize((224, 224), Image.BICUBIC)
            src_img = load_rgb(sp).resize((224, 224), Image.BICUBIC)
            gen_t = T.ToTensor()(gen_img).unsqueeze(0).to(device)
            src_t = T.ToTensor()(src_img).unsqueeze(0).to(device)
            # LPIPS expects [-1, 1] range
            gen_t = gen_t * 2 - 1
            src_t = src_t * 2 - 1
            gen_tensors.append(gen_t)
            src_tensors.append(src_t)
        gen_batch = torch.cat(gen_tensors, dim=0)
        src_batch = torch.cat(src_tensors, dim=0)
        dist = lpips_fn(gen_batch, src_batch).cpu().squeeze().tolist()
        if isinstance(dist, float):
            dist = [dist]
        results.extend(dist)
    return results


# --- Precompute CLIP style reference features from TEST dir ---
print("\n=== Precomputing CLIP style reference features from TEST dir ===")
style_ref_clip = {}
for style in STYLES:
    style_dir = TEST_DIR / style
    ref_paths = image_paths(style_dir)[:30]
    feats = extract_clip_image_features(ref_paths)
    style_ref_clip[style] = feats
    print(f"  {style}: {len(ref_paths)} refs")


# --- Build source image map (to find source path for LPIPS) ---
# SaMST generated images named: {src_style}__{src_stem}__to__{tgt_style}.png
# Source images are from TRAIN_DIR/{src_style}/{src_stem}.jpg
print("\n=== Building source image map ===")
src_path_map = {}  # (src_style, src_stem) -> path
for style in STYLES:
    style_dir = TRAIN_DIR / style
    for p in image_paths(style_dir):
        src_path_map[(style, p.stem)] = p
print(f"  {len(src_path_map)} source images indexed")


# --- Compute CLIP-S and LPIPS for each epoch ---
print("\n=== Computing CLIP-S and LPIPS per epoch ===")
results = []

for epoch in EPOCHS:
    epoch_dir = IMG_DIR / f"epoch_{epoch:02d}" / "images"
    if not epoch_dir.exists():
        print(f"  epoch {epoch}: missing, skipping")
        continue
    gen_paths = sorted([p for p in epoch_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS])
    t0 = time.time()

    # Extract CLIP features for generated images
    gen_clip_feats = extract_clip_image_features(gen_paths)

    # Compute CLIP-S (max cosine with style refs)
    clip_s_scores = []
    src_paths_for_lpips = []
    for feat, p in zip(gen_clip_feats, gen_paths):
        parts = p.stem.split("__to__")
        if len(parts) != 2 or parts[1] not in style_ref_clip:
            continue
        tgt_style = parts[1]
        src_style, src_stem = parts[0].split("__", 1)
        ref_feats = style_ref_clip[tgt_style].to(device)
        feat_dev = feat.to(device)
        cos_sim = (feat_dev.unsqueeze(0) @ ref_feats.T).squeeze(0)
        max_cos = cos_sim.max().item()
        clip_s_scores.append(max_cos)

        # Find source path for LPIPS
        sp = src_path_map.get((src_style, src_stem))
        if sp:
            src_paths_for_lpips.append(sp)
        else:
            src_paths_for_lpips.append(None)

    # Compute LPIPS (only for images where we have source paths)
    valid_indices = [i for i, sp in enumerate(src_paths_for_lpips) if sp is not None]
    valid_gen = [gen_paths[i] for i in valid_indices]
    valid_src = [src_paths_for_lpips[i] for i in valid_indices]

    if valid_gen:
        lpips_scores = compute_lpips_batch(valid_gen, valid_src)
    else:
        lpips_scores = []

    avg_clip_s = sum(clip_s_scores) / len(clip_s_scores) if clip_s_scores else 0
    avg_lpips = sum(lpips_scores) / len(lpips_scores) if lpips_scores else 0
    elapsed = time.time() - t0
    print(f"  epoch {epoch}: CLIP-S={avg_clip_s:.6f}, LPIPS={avg_lpips:.6f}  ({len(clip_s_scores)} CLIP, {len(lpips_scores)} LPIPS, {elapsed:.1f}s)")
    results.append({"epoch": epoch, "clip_s": avg_clip_s, "lpips": avg_lpips, "n_clip": len(clip_s_scores), "n_lpips": len(lpips_scores)})


# --- Save ---
out_csv = OUT_DIR / "samst_curve_clip_lpips.csv"
out_json = OUT_DIR / "samst_curve_clip_lpips.json"
with open(out_csv, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["epoch", "clip_s", "lpips", "n_clip", "n_lpips"])
    for r in results:
        writer.writerow([r["epoch"], f"{r['clip_s']:.6f}", f"{r['lpips']:.6f}", r["n_clip"], r["n_lpips"]])
with open(out_json, "w") as f:
    json.dump(results, f, indent=2)
print(f"\nSaved to {out_csv} and {out_json}")
print("Done.")
