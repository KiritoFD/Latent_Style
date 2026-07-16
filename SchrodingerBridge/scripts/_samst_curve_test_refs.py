"""Re-evaluate SaMST DINO-S using test dir as style refs (matching main table protocol)."""
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
IMG_DIR = pathlib.Path(r"I:\Github\Latent_Style\exp_samam\samst_curve_imgs")
OUT_DIR = pathlib.Path(r"I:\Github\Latent_Style\exp_samam\_dino_curve_repro")
OUT_DIR.mkdir(parents=True, exist_ok=True)

CACHE_DIR = r"I:\Github\Latent_Style\WEAVE\eval_cache"
MODEL_NAME = "facebook/dinov2-small"
BATCH_SIZE = 64
MAX_STYLE_REFS = 30

STYLES = ["Early_Renaissance", "Impressionism", "Minimalism", "Rococo", "Ukiyo_e"]
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
EPOCHS = [5, 10, 15]

DINO_TRANSFORM = T.Compose([
    T.Resize(224, interpolation=Image.BICUBIC),
    T.CenterCrop(224),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")


def load_rgb(path):
    with Image.open(path) as img:
        return img.convert("RGB")


def image_paths(root):
    return sorted(p for p in root.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS)


# --- Load DINOv2 ---
print("=== Loading DINOv2 ViT-S/14 ===")
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


# --- Precompute style reference features from TEST dir ---
print("\n=== Precomputing style reference features from TEST dir ===")
style_ref_features = {}
for style in STYLES:
    style_dir = TEST_DIR / style
    ref_paths = image_paths(style_dir)[:MAX_STYLE_REFS]
    feats = extract_cls_features(ref_paths)
    style_ref_features[style] = feats
    print(f"  {style}: {len(ref_paths)} refs")


# --- Compute DINO-S for each epoch ---
print("\n=== Computing DINO-S per epoch (test dir as style refs) ===")
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


# --- Save ---
out_csv = OUT_DIR / "samst_curve_test_refs.csv"
out_json = OUT_DIR / "samst_curve_test_refs.json"
with open(out_csv, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["epoch", "dino_s", "n_images"])
    for r in results:
        writer.writerow([r["epoch"], f"{r['dino_s']:.6f}", r["n_images"]])
with open(out_json, "w") as f:
    json.dump(results, f, indent=2)
print(f"\nSaved to {out_csv} and {out_json}")
print(f"Main table SaMST DINO-S = 0.2710")
if results:
    last = results[-1]
    print(f"Epoch 15 DINO-S = {last['dino_s']:.6f}")
print("Done.")
