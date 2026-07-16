"""Regenerate SaMST curve with paper-canonical protocol.

SaMST is per-style trained: each style has its own .model checkpoint.
For each epoch (5, 10, 15), load 5 style models and generate 750 images
(150 src x 5 target styles), then compute DINO-S using train dir as style refs.
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

# --- Paths (Windows, remote) ---
SAMST_REPO = pathlib.Path(r"C:\Users\Administrator\samst_repo")
sys.path.insert(0, str(SAMST_REPO))
from networks.transfer_net import TransformerNet  # noqa: E402

DATA_DIR = pathlib.Path(r"I:\datasets\wikiart_distinct5_samam_512_classview\train")
CKPT_ROOT = pathlib.Path(r"C:\Users\Administrator\samst_ckpts")
OUT_DIR = pathlib.Path(r"I:\Github\Latent_Style\exp_samam\_dino_curve_repro")
OUT_DIR.mkdir(parents=True, exist_ok=True)
IMG_DIR = pathlib.Path(r"I:\Github\Latent_Style\exp_samam\samst_curve_imgs")
IMG_DIR.mkdir(parents=True, exist_ok=True)

CACHE_DIR = r"I:\Github\Latent_Style\WEAVE\eval_cache"
MODEL_NAME = "facebook/dinov2-small"
BATCH_SIZE = 64
MAX_STYLE_REFS = 30
IMAGE_SIZE = 512

STYLES = ["Early_Renaissance", "Impressionism", "Minimalism", "Rococo", "Ukiyo_e"]
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
EPOCHS = [5, 10, 15]

DINO_TRANSFORM = T.Compose([
    T.Resize(224, interpolation=Image.BICUBIC),
    T.CenterCrop(224),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# SaMST input: ToTensor -> mul(255), so [0, 255] range
SAMST_TRANSFORM = T.Compose([
    T.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    T.ToTensor(),
    T.Lambda(lambda x: x.mul(255)),
])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")


def load_rgb(path):
    with Image.open(path) as img:
        return img.convert("RGB")


def image_paths(root):
    return sorted(p for p in root.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS)


# --- Select source images (same logic as SaMam: seed 42, 30 src/style) ---
print("=== Selecting source images ===")
sources = []  # (src_style, path)
for style in STYLES:
    style_dir = DATA_DIR / style
    paths = image_paths(style_dir)
    rng = random.Random(42)
    selected = paths[:]
    rng.shuffle(selected)
    selected = selected[:MAX_STYLE_REFS]
    for p in selected:
        sources.append((style, p))
print(f"  {len(sources)} sources (5 styles x 30)")


# --- Step 1: Generate images for each epoch ---
for epoch in EPOCHS:
    epoch_dir = IMG_DIR / f"epoch_{epoch:02d}" / "images"
    if epoch_dir.exists() and len(list(epoch_dir.glob("*.png"))) >= 700:
        print(f"epoch {epoch}: already has {len(list(epoch_dir.glob('*.png')))} images, skipping")
        continue

    print(f"\n=== Generating epoch_{epoch} images ===")
    epoch_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    n = 0

    for tgt_style in STYLES:
        ckpt_path = CKPT_ROOT / tgt_style / f"epoch_{epoch}.model"
        if not ckpt_path.exists():
            print(f"  WARNING: {ckpt_path} not found, skipping {tgt_style}")
            continue

        model = TransformerNet(style_num=1)
        state_dict = torch.load(str(ckpt_path), map_location=device)
        model.load_state_dict(state_dict)
        model = model.to(device).eval()

        with torch.inference_mode():
            for src_style, src_path in sources:
                content = SAMST_TRANSFORM(load_rgb(src_path)).unsqueeze(0).to(device)
                # style_id=[1] is the actual stylization (style_id=[0] is AE/identity)
                output, _ = model(content, style_id=[1])
                output = output.clamp(0, 255).cpu()
                name = f"{src_style}__{src_path.stem}__to__{tgt_style}.png"
                save_image(output[0] / 255.0, epoch_dir / name)
                n += 1

        del model
        torch.cuda.empty_cache()
        print(f"  {tgt_style}: done ({n} total)")

    print(f"  Generated {n} images in {time.time()-t0:.1f}s")


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


# --- Step 3: Precompute style reference features from train dir ---
print("\n=== Precomputing style reference features ===")
style_ref_features = {}
for style in STYLES:
    style_dir = DATA_DIR / style
    ref_paths = image_paths(style_dir)[:MAX_STYLE_REFS]
    feats = extract_cls_features(ref_paths)
    style_ref_features[style] = feats
    print(f"  {style}: {len(ref_paths)} refs")


# --- Step 4: Compute DINO-S for each epoch ---
print("\n=== Computing DINO-S per epoch ===")
results = []

for epoch in EPOCHS:
    epoch_dir = IMG_DIR / f"epoch_{epoch:02d}" / "images"
    if not epoch_dir.exists():
        print(f"  epoch {epoch}: missing, skipping")
        continue
    gen_paths = sorted([p for p in epoch_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS])
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
        print(f"  epoch {epoch}: no valid scores")
        continue
    avg_dino_s = sum(s[1] for s in dino_scores) / len(dino_scores)
    elapsed = time.time() - t0
    print(f"  epoch {epoch}: DINO-S={avg_dino_s:.6f}  ({len(dino_scores)} imgs, {elapsed:.1f}s)")
    results.append({"epoch": epoch, "dino_s": avg_dino_s, "n_images": len(dino_scores)})


# --- Step 5: Save ---
out_csv = OUT_DIR / "samst_curve_repro.csv"
out_json = OUT_DIR / "samst_curve_repro.json"
with open(out_csv, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["epoch", "dino_s", "n_images"])
    for r in results:
        writer.writerow([r["epoch"], f"{r['dino_s']:.6f}", r["n_images"]])
with open(out_json, "w") as f:
    json.dump(results, f, indent=2)
print(f"\nSaved to {out_csv} and {out_json}")
print(f"Total results: {len(results)}")
print("Done.")
