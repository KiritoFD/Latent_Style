"""Evaluate StyleAligned and Z-STAR baselines on D5, P2A, R5 datasets.

Computes CLIP-S (ViT-B/32), LPIPS (AlexNet), and MUSIQ per dataset using the
same protocol as the paper's main table. Self-contained — no src/ imports.

Usage (on remote RTX 3060):
    python tools/_eval_remote_baselines.py --method stylealigned --datasets D5,P2A,R5
    python tools/_eval_remote_baselines.py --method zstar --datasets D5,P2A,R5
    python tools/_eval_remote_baselines.py --method stylealigned --datasets D5 --batch_size 8
"""

import argparse
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

# ──────────────────────────────────────────────────────────────────
#  Dataset & path configuration
# ──────────────────────────────────────────────────────────────────

D5_STYLES = ["Early_Renaissance", "Impressionism", "Minimalism", "Rococo", "Ukiyo_e"]
P2A_STYLES = ["cezanne", "Hayao", "monet", "photo", "vangogh"]
R5_STYLES = ["Cubism", "Expressionism", "Pop_Art", "Romanticism", "Symbolism"]

DATASET_TEST_DIRS = {
    "D5": r"I:/datasets/wikiarts20_512_test",
    "P2A": r"I:/datasets/legacy256_overfit50/test",
    "R5": r"I:/datasets/wikiarts20_512_test",  # same base, different styles
}

METHOD_IMG_DIRS = {
    "stylealigned": {
        "D5": r"I:/Github/Latent_Style/SchrodingerBridge/exp/baseline_stylealigned/distinct5/images",
        "P2A": r"I:/Github/Latent_Style/SchrodingerBridge/exp/baseline_stylealigned/photo2art256/images",
        "R5": r"I:/Github/Latent_Style/SchrodingerBridge/exp/baseline_stylealigned/random5/images",
    },
    "zstar": {
        "D5": r"I:/Github/Latent_Style/SchrodingerBridge/exp/baseline_zstar/D5/images",
        "P2A": r"I:/Github/Latent_Style/SchrodingerBridge/exp/baseline_zstar/P2A/images",
        "R5": r"I:/Github/Latent_Style/SchrodingerBridge/exp/baseline_zstar/R5/images",
    },
}

DEVICE = "cuda"


# ──────────────────────────────────────────────────────────────────
#  Source-file resolution
# ──────────────────────────────────────────────────────────────────

def _find_source_file(search_dirs, stem):
    """Search directories for a source image matching *stem*."""
    for d in search_dirs:
        if not d.exists():
            continue
        for ext in (".jpg", ".jpeg", ".png"):
            candidate = d / (stem + ext)
            if candidate.exists():
                return candidate
    return None


def collect_pairs_wikiart(img_dir, test_dir, styles):
    """Collect (gen_path, src_path, tgt_style) for D5 / R5 datasets.

    Generated filename:  Style__Style__artist_title__to__TargetStyle.png
    Source file:          test_dir/Style/Style__artist_title.jpg
    """
    img_dir = Path(img_dir)
    test_dir = Path(test_dir)
    pairs = []
    missing = 0
    for gen_file in sorted(img_dir.glob("*.png")):
        stem = gen_file.stem
        if "__to__" not in stem:
            continue
        left, tgt_style = stem.rsplit("__to__", 1)
        parts = left.split("__", 2)
        if len(parts) < 3:
            continue
        src_style = parts[0]
        src_stem = parts[0] + "__" + parts[2]  # Style__artist_title
        src_path = _find_source_file([test_dir / src_style], src_stem)
        if src_path is None:
            missing += 1
            continue
        pairs.append((gen_file, src_path, tgt_style))
    if missing:
        print(f"    WARNING: {missing} generated images had no matching source")
    return pairs


def collect_pairs_p2a(img_dir, test_dir, styles):
    """Collect (gen_path, src_path, tgt_style) for P2A dataset.

    Generated filename:  Style__number__to__TargetStyle.png
    Source file:          test_dir/Style/number.jpg
    """
    img_dir = Path(img_dir)
    test_dir = Path(test_dir)
    pairs = []
    missing = 0
    for gen_file in sorted(img_dir.glob("*.png")):
        stem = gen_file.stem
        if "__to__" not in stem:
            continue
        left, tgt_style = stem.rsplit("__to__", 1)
        parts = left.split("__", 1)
        if len(parts) < 2:
            continue
        src_style = parts[0]
        number = parts[1]
        src_path = _find_source_file([test_dir / src_style], number)
        if src_path is None:
            missing += 1
            continue
        pairs.append((gen_file, src_path, tgt_style))
    if missing:
        print(f"    WARNING: {missing} generated images had no matching source")
    return pairs


# ──────────────────────────────────────────────────────────────────
#  CLIP-S computation (batched, matching paper protocol)
# ──────────────────────────────────────────────────────────────────

@torch.no_grad()
def _extract_clip_features_batch(paths, clip_model, clip_proc, batch_size):
    """Extract and L2-normalize CLIP image features in batches."""
    all_feats = []
    for i in tqdm(range(0, len(paths), batch_size), desc="    CLIP feats", leave=False):
        batch_paths = paths[i : i + batch_size]
        imgs = [Image.open(p).convert("RGB") for p in batch_paths]
        inputs = clip_proc(images=imgs, return_tensors="pt", padding=True).to(DEVICE)
        out = clip_model.get_image_features(**inputs)
        if isinstance(out, torch.Tensor):
            feats = out
        elif hasattr(out, "image_embeds"):
            feats = out.image_embeds
        else:
            vision_out = clip_model.vision_model(pixel_values=inputs["pixel_values"])
            feats = clip_model.visual_projection(vision_out[1])
        # L2-normalize each vector
        feats = feats / feats.norm(p=2, dim=-1, keepdim=True).clamp(min=1e-8)
        all_feats.append(feats.cpu().numpy())
    return np.concatenate(all_feats, axis=0)


def compute_clip_style(gen_paths, prototype_map, gen_feats, styles, pairs):
    """Compute per-style and overall CLIP-S.

    prototype_map: tgt_style -> normalized prototype [1, D]
    gen_feats:     [N, D] normalized features for all generated images
    pairs:         list of (gen_path, src_path, tgt_style)
    """
    per_style = {}
    for tgt_style in styles:
        proto = prototype_map.get(tgt_style)
        if proto is None:
            continue
        indices = [i for i, (_, _, t) in enumerate(pairs) if t == tgt_style]
        if not indices:
            continue
        style_feats = gen_feats[indices]  # [n, D]
        sims = (style_feats * proto).sum(axis=1)
        per_style[tgt_style] = float(sims.mean())

    # Overall: each generated image compared to its target-style prototype
    all_sims = []
    for i, (_, _, tgt_style) in enumerate(pairs):
        proto = prototype_map.get(tgt_style)
        if proto is not None:
            all_sims.append(float((gen_feats[i] * proto).sum()))
    overall = float(np.mean(all_sims)) if all_sims else 0.0
    return overall, per_style


def build_prototypes(test_dir, styles, clip_model, clip_proc, batch_size):
    """Build style prototypes: mean of normalized CLIP features, then normalize.

    This matches the paper protocol (and run_evaluation.py lines 3738-3744):
      1. Extract CLIP features for all images in a style directory
      2. L2-normalize each feature vector
      3. Take the mean across all vectors
      4. L2-normalize the mean
    """
    test_dir = Path(test_dir)
    prototype_map = {}
    for style in styles:
        style_dir = test_dir / style
        if not style_dir.exists():
            print(f"    SKIP prototype for {style}: directory not found")
            continue
        files = sorted(
            f for f in style_dir.iterdir()
            if f.suffix.lower() in {".jpg", ".jpeg", ".png"}
        )
        if not files:
            print(f"    SKIP prototype for {style}: no images found")
            continue
        feats = _extract_clip_features_batch(files, clip_model, clip_proc, batch_size)
        # feats are already L2-normalized per vector (done in extraction)
        proto = feats.mean(axis=0, keepdims=True)
        proto = proto / np.linalg.norm(proto, axis=1, keepdims=True).clip(min=1e-8)
        prototype_map[style] = proto
        print(f"    Prototype {style}: {len(files)} ref images")
    return prototype_map


# ──────────────────────────────────────────────────────────────────
#  LPIPS computation
# ──────────────────────────────────────────────────────────────────

@torch.no_grad()
def compute_lpips(gen_paths, src_paths, lpips_metric):
    """Mean LPIPS between generated and source (source resized to gen dims)."""
    vals = []
    for gp, sp in tqdm(zip(gen_paths, src_paths), total=len(gen_paths),
                       desc="    LPIPS", leave=False):
        gen = Image.open(gp).convert("RGB")
        src = Image.open(sp).convert("RGB")
        w, h = gen.size
        src = src.resize((w, h), Image.LANCZOS)
        gen_t = (torch.from_numpy(np.array(gen).transpose(2, 0, 1)).float()
                 .unsqueeze(0).to(DEVICE) / 255.0)
        src_t = (torch.from_numpy(np.array(src).transpose(2, 0, 1)).float()
                 .unsqueeze(0).to(DEVICE) / 255.0)
        v = lpips_metric(gen_t, src_t).item()
        vals.append(v)
    return float(np.mean(vals))


# ──────────────────────────────────────────────────────────────────
#  MUSIQ computation
# ──────────────────────────────────────────────────────────────────

@torch.no_grad()
def compute_musiq(gen_paths, musiq_metric):
    """Mean MUSIQ (no-reference IQA) over generated images."""
    vals = []
    for p in tqdm(gen_paths, desc="    MUSIQ", leave=False):
        img = Image.open(p).convert("RGB")
        t = (torch.from_numpy(np.array(img).transpose(2, 0, 1)).float()
             .unsqueeze(0).to(DEVICE) / 255.0)
        v = musiq_metric(t).item()
        vals.append(v)
    return float(np.mean(vals))


# ──────────────────────────────────────────────────────────────────
#  Per-dataset evaluation driver
# ──────────────────────────────────────────────────────────────────

def eval_dataset(pairs, test_dir, styles, label, clip_model, clip_proc,
                 lpips_metric, musiq_metric, batch_size):
    """Run CLIP-S, LPIPS, MUSIQ on one dataset. Returns dict of results."""
    test_dir = Path(test_dir)

    if not pairs:
        print(f"  [{label}] No pairs found, skipping.")
        return None

    gen_paths = [g for g, s, t in pairs]
    src_paths = [s for g, s, t in pairs]

    print(f"  [{label}] {len(pairs)} valid pairs")

    # ── Build style prototypes ──
    print(f"  [{label}] Building style prototypes ...")
    prototype_map = build_prototypes(test_dir, styles, clip_model, clip_proc,
                                     batch_size)

    # ── Extract generated-image CLIP features ──
    print(f"  [{label}] Extracting CLIP features for generated images ...")
    gen_feats = _extract_clip_features_batch(gen_paths, clip_model, clip_proc,
                                             batch_size)

    # ── CLIP-S ──
    overall_cs, per_style_cs = compute_clip_style(
        gen_paths, prototype_map, gen_feats, styles, pairs
    )
    print(f"  [{label}] CLIP-S = {overall_cs:.4f}")
    for style, val in sorted(per_style_cs.items()):
        n = sum(1 for _, _, t in pairs if t == style)
        print(f"    {style}: {val:.4f}  ({n} images)")

    # ── LPIPS ──
    print(f"  [{label}] Computing LPIPS ...")
    overall_lp = compute_lpips(gen_paths, src_paths, lpips_metric)
    print(f"  [{label}] LPIPS = {overall_lp:.4f}")

    # ── MUSIQ ──
    overall_musiq = None
    if musiq_metric is not None:
        print(f"  [{label}] Computing MUSIQ ...")
        overall_musiq = compute_musiq(gen_paths, musiq_metric)
        print(f"  [{label}] MUSIQ = {overall_musiq:.4f}")

    style_counts = Counter(t for _, _, t in pairs)
    return {
        "clip_style": overall_cs,
        "per_style_clip": per_style_cs,
        "lpips": overall_lp,
        "musiq": overall_musiq,
        "n_pairs": len(pairs),
        "style_counts": dict(style_counts),
    }


# ──────────────────────────────────────────────────────────────────
#  Main
# ──────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate StyleAligned / Z-STAR baselines on D5, P2A, R5"
    )
    parser.add_argument(
        "--method", required=True, choices=["stylealigned", "zstar"],
        help="Which method to evaluate",
    )
    parser.add_argument(
        "--datasets", default="D5,P2A,R5",
        help="Comma-separated list of datasets (D5, P2A, R5)",
    )
    parser.add_argument(
        "--batch_size", type=int, default=16,
        help="Batch size for CLIP feature extraction (default: 16)",
    )
    args = parser.parse_args()

    datasets = [d.strip() for d in args.datasets.split(",")]
    method = args.method
    batch_size = args.batch_size

    print("=" * 64)
    print(f"  Baseline Evaluation: {method.upper()}")
    print(f"  Datasets: {datasets}")
    print(f"  Batch size: {batch_size}")
    print(f"  Device: {DEVICE}")
    print("=" * 64)

    # ── Load metrics (once, reuse across datasets) ──
    print("\nLoading CLIP ViT-B/32 ...")
    from transformers import CLIPModel, CLIPProcessor
    clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(DEVICE).eval()
    clip_proc = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    print("Loading LPIPS (AlexNet) ...")
    import pyiqa
    lpips_metric = pyiqa.create_metric("lpips", device=DEVICE)

    musiq_metric = None
    try:
        print("Loading MUSIQ ...")
        musiq_metric = pyiqa.create_metric("musiq", device=DEVICE)
    except Exception as e:
        print(f"  MUSIQ unavailable: {e}")

    # ── Evaluate each dataset ──
    results = {}
    for ds_name in datasets:
        ds_styles = {"D5": D5_STYLES, "P2A": P2A_STYLES, "R5": R5_STYLES}.get(ds_name)
        if ds_styles is None:
            print(f"\n  Unknown dataset '{ds_name}', skipping.")
            continue

        test_dir = DATASET_TEST_DIRS[ds_name]
        img_dir = METHOD_IMG_DIRS[method][ds_name]

        if not Path(img_dir).exists():
            print(f"\n  [{ds_name}] Image dir not found: {img_dir}")
            continue
        if not Path(test_dir).exists():
            print(f"\n  [{ds_name}] Test dir not found: {test_dir}")
            continue

        print(f"\n{'─' * 60}")
        print(f"  Dataset: {ds_name}  |  Method: {method}")
        print(f"  Images:  {img_dir}")
        print(f"  Test:    {test_dir}")
        print(f"  Styles:  {ds_styles}")
        print(f"{'─' * 60}")

        # Collect pairs
        if ds_name in ("D5", "R5"):
            pairs = collect_pairs_wikiart(img_dir, test_dir, ds_styles)
        else:
            pairs = collect_pairs_p2a(img_dir, test_dir, ds_styles)

        result = eval_dataset(
            pairs, test_dir, ds_styles, ds_name,
            clip_model, clip_proc, lpips_metric, musiq_metric, batch_size,
        )
        if result is not None:
            results[ds_name] = result

    # ── Summary ──
    print("\n" + "=" * 64)
    print(f"  RESULTS SUMMARY — {method.upper()}")
    print("=" * 64)
    header = f"{'Dataset':<8} {'CLIP-S':>8} {'LPIPS':>8} {'MUSIQ':>8} {'N':>6}"
    print(header)
    print("-" * len(header))
    for ds_name, r in results.items():
        musiq_str = f"{r['musiq']:.4f}" if r["musiq"] is not None else "   n/a"
        print(f"{ds_name:<8} {r['clip_style']:8.4f} {r['lpips']:8.4f} {musiq_str:>8} {r['n_pairs']:6d}")
    print("=" * 64)

    # ── Per-style detail ──
    for ds_name, r in results.items():
        if r.get("per_style_clip"):
            print(f"\n  {ds_name} per-style CLIP-S:")
            sc = r["style_counts"]
            for style, val in sorted(r["per_style_clip"].items()):
                print(f"    {style:<22s} {val:.4f}  ({sc.get(style, 0)} images)")
    print()

    # Cleanup
    del clip_model
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
