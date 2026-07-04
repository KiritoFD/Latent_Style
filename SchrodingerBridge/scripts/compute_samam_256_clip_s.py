"""Compute CLIP-S for samam_256 with corrected path."""
import json
import sys
from pathlib import Path

# Reuse functions from compute_baseline_clip_s.py
sys.path.insert(0, "/mnt/c/Users/Administrator")
from compute_baseline_clip_s import (
    STYLE_NAMES, REF_ROOT, CLIP_CACHE,
    parse_tgt_style, collect_images, compute_clip_features,
)

import torch
import torch.nn.functional as F

GEN_DIR = "/mnt/i/Github/Latent_Style/exp_samam/eval_256/samam_final_20k_256/step_020000/images"
OUTPUT = "/mnt/i/exp_baseline_clip_s.json"

device = torch.device("cuda")
dtype = torch.float16

# Load existing results
results = json.loads(Path(OUTPUT).read_text())

# Load CLIP
from transformers import CLIPModel, CLIPProcessor
clip_src = CLIP_CACHE if Path(CLIP_CACHE).exists() else "openai/clip-vit-base-patch32"
print(f"[INFO] Loading CLIP from: {clip_src}")
clip_model = CLIPModel.from_pretrained(clip_src).to(device=device, dtype=dtype).eval()
clip_processor = CLIPProcessor.from_pretrained(clip_src)

# Compute ref prototypes
print("[INFO] Computing ref prototypes...")
ref_prototypes = {}
for style in STYLE_NAMES:
    ref_dir = Path(REF_ROOT) / style
    if not ref_dir.exists():
        continue
    ref_files = collect_images(ref_dir, max_images=50)
    ref_feats = compute_clip_features(ref_files, clip_model, clip_processor, device, dtype, 16)
    if ref_feats is not None:
        proto = ref_feats.mean(dim=0)
        proto = F.normalize(proto, dim=-1)
        ref_prototypes[style] = proto
        print(f"  {style}: {len(ref_files)} refs")

# Compute for samam_256
gen_path = Path(GEN_DIR)
gen_files = collect_images(gen_path, 750)
print(f"[samam_256] {len(gen_files)} images in {GEN_DIR}")

gen_feats = compute_clip_features(gen_files, clip_model, clip_processor, device, dtype, 16)

scores = []
for i, f in enumerate(gen_files):
    tgt = parse_tgt_style(f.name)
    if tgt is None or tgt not in ref_prototypes:
        continue
    proto = ref_prototypes[tgt].to(device)
    gf = gen_feats[i].to(device).to(dtype)
    sim = float(F.cosine_similarity(gf.unsqueeze(0), proto.unsqueeze(0)).item())
    scores.append(sim)

if scores:
    clip_s = sum(scores) / len(scores)
    results["samam_256"] = {
        "n_images": len(gen_files),
        "n_scored": len(scores),
        "clip_s": clip_s,
    }
    print(f"  CLIP-S: {clip_s:.4f} ({len(scores)} images)")

Path(OUTPUT).write_text(json.dumps(results, indent=2))
print(f"[INFO] Updated {OUTPUT}")
