"""Compute DINO-S for SaMam curve points using remote Windows Python.

Uses curve_eval_30src (real files, not symlinks) with DINOv2 ViT-S/14.
Matches the paper_canonical protocol from WEAVE/utils/compute_dino_metrics.py.
"""
import json
import os
import pathlib
import sys
import time

import torch
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image
from transformers import AutoModel

# --- Config ---
CURVE_BASE = pathlib.Path(
    r"I:\Github\Latent_Style\exp_samam\training"
    r"\samam_distinct5_512_scratch_7k_250eval_remote"
    r"\curve_eval_30src"
)
STYLE_REF_DIR = pathlib.Path(
    r"I:\datasets\wikiart_distinct5_samam_512_classview\train"
)
OUT_DIR = pathlib.Path(r"I:\Github\Latent_Style\exp_samam\_dino_curve")
OUT_DIR.mkdir(parents=True, exist_ok=True)

CACHE_DIR = r"I:\Github\Latent_Style\WEAVE\eval_cache"
MODEL_NAME = "facebook/dinov2-small"
BATCH_SIZE = 64
MAX_STYLE_REFS = 30

# Match paper_canonical transform exactly
DINO_TRANSFORM = T.Compose([
    T.Resize(224, interpolation=Image.BICUBIC),
    T.CenterCrop(224),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

STYLES = ["Early_Renaissance", "Impressionism", "Minimalism", "Rococo", "Ukiyo_e"]
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# Sample ~15 steps from available 28 steps + last
ALL_STEPS = sorted([int(d.name.split("_")[1]) for d in CURVE_BASE.glob("step_*")])
SAMPLE_STEPS = [250, 500, 1000, 2000, 3000, 4000, 5000, 6000, 7000]
# Add last if exists
LAST_DIR = CURVE_BASE / "last"
HAS_LAST = LAST_DIR.exists()

print(f"Available steps: {ALL_STEPS}")
print(f"Sampling: {SAMPLE_STEPS}")
print(f"Has last: {HAS_LAST}")

# Filter to existing steps
SAMPLE_STEPS = [s for s in SAMPLE_STEPS if s in ALL_STEPS]
print(f"Filtered sample steps: {SAMPLE_STEPS}")

# --- Device ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# --- Load model (match paper_canonical load_dino) ---
print("Loading DINOv2 ViT-S/14...")
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

# Try loading from local cache
repo_dir = pathlib.Path(CACHE_DIR) / f"models--facebook--dinov2-small"
snap_root = repo_dir / "snapshots"
if snap_root.exists():
    revisions = [p for p in snap_root.iterdir() if p.is_dir()]
    if revisions:
        local_path = str(revisions[0])
        print(f"Loading from local snapshot: {local_path}")
        model = AutoModel.from_pretrained(local_path).to(device).eval()
    else:
        print("No snapshot revisions found, trying cache_dir direct")
        model = AutoModel.from_pretrained(MODEL_NAME, cache_dir=CACHE_DIR).to(device).eval()
else:
    # Maybe the model files are directly in repo_dir
    print(f"Snapshot not found at {snap_root}")
    print(f"Repo dir contents: {list(repo_dir.iterdir()) if repo_dir.exists() else 'N/A'}")
    # Try with allow_network
    os.environ["HF_HUB_OFFLINE"] = "0"
    os.environ["TRANSFORMERS_OFFLINE"] = "0"
    model = AutoModel.from_pretrained(MODEL_NAME, cache_dir=CACHE_DIR).to(device).eval()

print("Model loaded.")


def load_image(path):
    with Image.open(path) as img:
        return img.convert("RGB")


@torch.inference_mode()
def extract_cls_features(paths, model, device, batch_size):
    """Extract CLS features, matching paper_canonical protocol."""
    cls_features = []
    for start in range(0, len(paths), batch_size):
        batch_paths = paths[start:start + batch_size]
        pixels = torch.stack([DINO_TRANSFORM(load_image(p)) for p in batch_paths]).to(device)
        output = model(pixels, output_hidden_states=True)
        cls = F.normalize(output.last_hidden_state[:, 0, :].float(), dim=-1).cpu()
        cls_features.append(cls)
    return torch.cat(cls_features, dim=0)  # [N, 384]


# --- Precompute style reference features ---
print("\nPrecomputing style reference features...")
style_ref_features = {}
for style in STYLES:
    style_dir = STYLE_REF_DIR / style
    if not style_dir.exists():
        print(f"  WARNING: {style_dir} not found, skipping")
        continue
    ref_paths = sorted([p for p in style_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS])
    ref_paths = ref_paths[:MAX_STYLE_REFS]
    feats = extract_cls_features(ref_paths, model, device, BATCH_SIZE)
    style_ref_features[style] = feats  # [N, 384]
    print(f"  {style}: {len(ref_paths)} refs, tensor {feats.shape}")

# --- Compute DINO-S for each step ---
print("\nComputing DINO-S per step...")
results = []


def compute_step_dino(step_dir, step_label):
    """Compute DINO-S for one step directory."""
    images_dir = step_dir / "images"
    gen_paths = sorted([p for p in images_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS])
    t0 = time.time()
    print(f"\n{step_label}: {len(gen_paths)} images")

    # Extract all generated features
    gen_feats = extract_cls_features(gen_paths, model, device, BATCH_SIZE)  # [M, 384]

    # Parse target styles and compute DINO-S per image
    dino_scores = []
    for i, (feat, p) in enumerate(zip(gen_feats, gen_paths)):
        parts = p.stem.split("__to__")
        tgt_style = parts[1] if len(parts) == 2 and parts[1] in style_ref_features else None
        if tgt_style is None:
            continue
        ref_feats = style_ref_features[tgt_style].to(device)  # [N, 384]
        feat_dev = feat.to(device)
        cos_sim = (feat_dev.unsqueeze(0) @ ref_feats.T).squeeze(0)  # [N]
        max_cos = cos_sim.max().item()
        dino_scores.append((tgt_style, max_cos))

    if dino_scores:
        avg_dino_s = sum(s[1] for s in dino_scores) / len(dino_scores)
        per_style = {}
        for style in STYLES:
            vals = [s[1] for s in dino_scores if s[0] == style]
            if vals:
                per_style[style] = sum(vals) / len(vals)
        elapsed = time.time() - t0
        print(f"  DINO-S: {avg_dino_s:.6f}  ({elapsed:.1f}s)")
        return {
            "step": step_label,
            "n_images": len(dino_scores),
            "dino_s": avg_dino_s,
            "per_style": per_style,
        }
    return None


for step in SAMPLE_STEPS:
    step_dir = CURVE_BASE / f"step_{step:06d}"
    if step_dir.exists():
        r = compute_step_dino(step_dir, step)
        if r:
            results.append(r)

# Add last
if HAS_LAST:
    r = compute_step_dino(LAST_DIR, "last")
    if r:
        results.append(r)

# --- Save ---
out_json = OUT_DIR / "dino_samam_curve.json"
out_csv = OUT_DIR / "dino_samam_curve.csv"
with open(out_json, "w") as f:
    json.dump(results, f, indent=2)
with open(out_csv, "w") as f:
    f.write("step,dino_s,n_images\n")
    for r in results:
        f.write(f"{r['step']},{r['dino_s']:.6f},{r['n_images']}\n")
print(f"\nSaved to {out_json} and {out_csv}")
print(f"Total results: {len(results)}")
print("Done.")