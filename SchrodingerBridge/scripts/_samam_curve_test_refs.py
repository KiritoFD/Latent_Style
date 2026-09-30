"""Re-evaluate DINO-S using test dir as style references (matching main table protocol).

Uses existing curve_eval_30src images but replaces style refs from train/ to test/.
If step 20000 DINO-S matches main table (0.4771), then only style refs were the issue.
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

# --- Paths ---
TEST_DIR = pathlib.Path(r"I:\datasets\wikiart_distinct5_512_images\test")  # main table style refs
CURVE_BASE = pathlib.Path(
    r"I:\Github\Latent_Style\exp_samam\training"
    r"\samam_distinct5_512_scratch_7k_250eval_remote\curve_eval_30src"
)
OUT_DIR = pathlib.Path(r"I:\Github\Latent_Style\exp_samam\_dino_curve_repro")
OUT_DIR.mkdir(parents=True, exist_ok=True)

CACHE_DIR = r"I:\Github\Latent_Style\WEAVE\eval_cache"
MODEL_NAME = "facebook/dinov2-small"
BATCH_SIZE = 64
MAX_STYLE_REFS = 30

STYLES = ["Early_Renaissance", "Impressionism", "Minimalism", "Rococo", "Ukiyo_e"]
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
SAMPLE_STEPS = [250, 500, 1000, 2000, 3000, 5000, 7000, 20000]

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
print(f"  test_dir: {TEST_DIR}")
style_ref_features = {}
for style in STYLES:
    style_dir = TEST_DIR / style
    ref_paths = image_paths(style_dir)[:MAX_STYLE_REFS]
    feats = extract_cls_features(ref_paths)
    style_ref_features[style] = feats
    print(f"  {style}: {len(ref_paths)} refs")


# --- Compute DINO-S for each step ---
print("\n=== Computing DINO-S per step (test dir as style refs) ===")
results = []

for step in SAMPLE_STEPS:
    step_dir = CURVE_BASE / f"step_{step:06d}" / "images"
    if not step_dir.exists():
        print(f"  step {step}: missing, skipping")
        continue
    gen_paths = sorted([p for p in step_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS])
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
        print(f"  step {step}: no valid scores")
        continue
    avg_dino_s = sum(s[1] for s in dino_scores) / len(dino_scores)
    elapsed = time.time() - t0
    print(f"  step {step}: DINO-S={avg_dino_s:.6f}  ({len(dino_scores)} imgs, {elapsed:.1f}s)")
    results.append({"step": step, "dino_s": avg_dino_s, "n_images": len(dino_scores)})


# --- Save ---
out_csv = OUT_DIR / "samam_curve_test_refs.csv"
out_json = OUT_DIR / "samam_curve_test_refs.json"
with open(out_csv, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["step", "dino_s", "n_images"])
    for r in results:
        writer.writerow([r["step"], f"{r['dino_s']:.6f}", r["n_images"]])
with open(out_json, "w") as f:
    json.dump(results, f, indent=2)
print(f"\nSaved to {out_csv} and {out_json}")
print(f"Main table SaMam DINO-S = 0.4771")
if results:
    last = results[-1]
    print(f"Step 20000 DINO-S = {last['dino_s']:.6f} (diff={0.4771 - last['dino_s']:.4f})")
print("Done.")
