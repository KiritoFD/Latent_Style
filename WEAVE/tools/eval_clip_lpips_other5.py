"""Compute CLIP-S and LPIPS metrics for a directory of generated images.

Generates a metrics.csv compatible with compute_dino_metrics.py.
Used for SaMam and SaMST outputs that don't have run_evaluation.py's full pipeline.

Filename convention (matches ours):
    {src_style}__{src_stem}__to__{tgt_style}.png

Usage:
    python tools/eval_clip_lpips_other5.py \
        --gen-dir <method>/step_000001/images \
        --test-dir <other5 test root> \
        --output-dir <method> \
        --style-names Abstract_Expressionism,Art_Nouveau_Modern,Cubism,Expressionism,Symbolism \
        --num-src 30
"""
import argparse
import csv
import json
import re
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
import torchvision.transforms as T
import lpips
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
STYLE_NAMES_DEFAULT = [
    "Abstract_Expressionism",
    "Art_Nouveau_Modern",
    "Cubism",
    "Expressionism",
    "Symbolism",
]


def parse_filename(name: str):
    """Parse {src_style}__{src_stem}__to__{tgt_style}.png

    Handles double-style-prefix case where src_stem starts with src_style
    (e.g., Cubism__Cubism__artist_name__to__Cubism.png).
    Uses rsplit to find the LAST __to__ separator, then splits the left part
    at the FIRST __ to get src_style and src_stem.
    """
    stem = Path(name).stem
    if "__to__" not in stem:
        return None
    left, tgt = stem.rsplit("__to__", 1)
    # Split at first __ to get src_style and src_stem
    if "__" not in left:
        return None
    parts = left.split("__", 1)
    return {
        "src_style": parts[0],
        "src_stem": parts[1],
        "tgt_style": tgt,
    }


def load_image_01(path: Path, size: int = 224) -> torch.Tensor:
    img = Image.open(path).convert("RGB").resize((size, size), Image.BICUBIC)
    return T.ToTensor()(img)


def load_image_for_lpips(path: Path, size: int = 224) -> torch.Tensor:
    """LPIPS expects [-1, 1] range."""
    img = Image.open(path).convert("RGB").resize((size, size), Image.BICUBIC)
    t = T.ToTensor()(img)
    return t * 2.0 - 1.0


def get_clip_image_features(clip_model, inputs):
    """Get normalized CLIP image features, handling different transformers versions."""
    feat = clip_model.get_image_features(**inputs)
    # Some transformers versions return BaseModelOutputWithPooling instead of tensor
    if not isinstance(feat, torch.Tensor):
        feat = feat.pooler_output if hasattr(feat, 'pooler_output') else feat[0]
    return F.normalize(feat.float(), dim=-1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gen-dir", required=True, help="Directory of generated images")
    parser.add_argument("--test-dir", required=True, help="Test dataset root with style subdirs")
    parser.add_argument("--output-dir", required=True, help="Output dir for metrics.csv")
    parser.add_argument("--style-names", default=",".join(STYLE_NAMES_DEFAULT))
    parser.add_argument("--num-src", type=int, default=30)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--clip-model-name", default="openai/clip-vit-base-patch32")
    parser.add_argument("--clip-cache-dir", default="")
    parser.add_argument("--clip-local-dir", default="G:/GitHub/Latent_Style/eval_cache/manual_clip/openai-clip-vit-base-patch32")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--image-size", type=int, default=224)
    args = parser.parse_args()

    style_names = [s.strip() for s in args.style_names.split(",") if s.strip()]
    device = torch.device(args.device)
    gen_dir = Path(args.gen_dir)
    test_dir = Path(args.test_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Collect generated images
    gen_files = sorted([p for p in gen_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS])
    print(f"[INFO] Found {len(gen_files)} generated images in {gen_dir}")

    # Parse filenames and build rows
    rows = []
    for gf in gen_files:
        parsed = parse_filename(gf.name)
        if parsed is None:
            print(f"[WARN] Cannot parse: {gf.name}")
            continue
        # Find source image
        src_style = parsed["src_style"]
        src_stem = parsed["src_stem"]
        src_dir = test_dir / src_style
        src_path = None
        for ext in IMAGE_EXTS:
            candidate = src_dir / f"{src_stem}{ext}"
            if candidate.exists():
                src_path = candidate
                break
        if src_path is None:
            # Try with .jpg extension only
            print(f"[WARN] Source not found for {gf.name} (src_stem={src_stem})")
            continue
        rows.append({
            "src_style": src_style,
            "tgt_style": parsed["tgt_style"],
            "src_image": src_path.name,
            "gen_image": gf.name,
            "src_path": str(src_path),
            "gen_path": str(gf),
        })
    print(f"[INFO] {len(rows)} valid (src, gen) pairs")

    # Load CLIP (prefer local dir to avoid network)
    local_dir = Path(args.clip_local_dir) if args.clip_local_dir else None
    if local_dir and local_dir.exists():
        print(f"[INFO] Loading CLIP from local dir: {local_dir}")
        clip_model = CLIPModel.from_pretrained(str(local_dir)).to(device).eval()
        clip_processor = CLIPProcessor.from_pretrained(str(local_dir))
    else:
        cache_dir = args.clip_cache_dir.strip() or None
        clip_model = CLIPModel.from_pretrained(args.clip_model_name, cache_dir=cache_dir).to(device).eval()
        clip_processor = CLIPProcessor.from_pretrained(args.clip_model_name, cache_dir=cache_dir)
    print(f"[INFO] CLIP loaded")

    # Load LPIPS
    lpips_fn = lpips.LPIPS(net="alex", verbose=False).to(device).eval()
    print("[INFO] LPIPS loaded (alex)")

    # Precompute style reference CLIP prototypes (mean of normalized features, then re-normalize)
    # This matches run_evaluation.py's ref_clip_prototypes definition.
    style_ref_prototypes = {}
    style_ref_matrices = {}
    for sname in style_names:
        sdir = test_dir / sname
        ref_files = sorted([p for p in sdir.iterdir() if p.suffix.lower() in IMAGE_EXTS])[:args.num_src]
        if not ref_files:
            continue
        feats = []
        for rf in ref_files:
            img = Image.open(rf).convert("RGB")
            inputs = clip_processor(images=img, return_tensors="pt").to(device)
            with torch.no_grad():
                feat = get_clip_image_features(clip_model, inputs)
            feats.append(feat.cpu())
        stacked = torch.cat(feats, dim=0)
        stacked = F.normalize(stacked, dim=-1)
        proto = stacked.mean(dim=0, keepdim=True)
        proto = F.normalize(proto, dim=-1)
        style_ref_prototypes[sname] = proto.to(device)
        style_ref_matrices[sname] = stacked
        print(f"  Style refs: {sname} -> {len(ref_files)} images (prototype shape={tuple(proto.shape)})")

    # Compute metrics
    t0 = time.time()
    results = []
    for i, row in enumerate(rows):
        gen_img = Image.open(row["gen_path"]).convert("RGB")
        src_img = Image.open(row["src_path"]).convert("RGB")

        # CLIP
        gen_inputs = clip_processor(images=gen_img, return_tensors="pt").to(device)
        src_inputs = clip_processor(images=src_img, return_tensors="pt").to(device)
        with torch.no_grad():
            gen_feat = get_clip_image_features(clip_model, gen_inputs).cpu()
            src_feat = get_clip_image_features(clip_model, src_inputs).cpu()

        # CLIP-S: cosine with target style prototype (matches run_evaluation.py)
        tgt_style = row["tgt_style"]
        if tgt_style in style_ref_prototypes:
            clip_s = float(F.cosine_similarity(gen_feat, style_ref_prototypes[tgt_style].cpu(), dim=-1).item())
        else:
            clip_s = 0.0

        # LPIPS
        gen_t = load_image_for_lpips(Path(row["gen_path"]), args.image_size).unsqueeze(0).to(device)
        src_t = load_image_for_lpips(Path(row["src_path"]), args.image_size).unsqueeze(0).to(device)
        with torch.no_grad():
            lpips_val = float(lpips_fn(gen_t, src_t).item())

        results.append({
            "src_style": row["src_style"],
            "tgt_style": row["tgt_style"],
            "src_image": row["src_image"],
            "gen_image": row["gen_image"],
            "content_lpips": lpips_val,
            "clip_style": clip_s,
        })

        if (i + 1) % 100 == 0:
            print(f"  Processed {i+1}/{len(rows)}...")

    elapsed = time.time() - t0
    print(f"[INFO] Evaluated {len(results)} pairs in {elapsed:.1f}s")

    # Save metrics.csv
    csv_path = out_dir / "metrics.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["src_style", "tgt_style", "src_image", "gen_image", "content_lpips", "clip_dir", "clip_style"])
        for r in results:
            writer.writerow([r["src_style"], r["tgt_style"], r["src_image"], "images/" + r["gen_image"],
                         r["content_lpips"], 0.0, r["clip_style"]])
    print(f"[INFO] Saved {csv_path}")

    # Save summary
    all_clip_s = sum(r["clip_style"] for r in results) / max(1, len(results))
    all_lpips = sum(r["content_lpips"] for r in results) / max(1, len(results))
    off_indices = [i for i, r in enumerate(results) if r["src_style"] != r["tgt_style"]]
    off_clip_s = sum(results[i]["clip_style"] for i in off_indices) / max(1, len(off_indices))
    off_lpips = sum(results[i]["content_lpips"] for i in off_indices) / max(1, len(off_indices))

    # Per-style CLIP-S
    per_style = {}
    for sname in style_names:
        style_rows = [r for r in results if r["tgt_style"] == sname]
        if style_rows:
            per_style[sname] = sum(r["clip_style"] for r in style_rows) / len(style_rows)

    summary = {
        "n_all": len(results),
        "n_off_diagonal": len(off_indices),
        "clip_style_global": all_clip_s,
        "lpips_global": all_lpips,
        "off_clip_style": off_clip_s,
        "off_lpips": off_lpips,
        "clip_style_eval_identity_by_target_style": per_style,
    }
    summary_path = out_dir / "clip_lpips_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
