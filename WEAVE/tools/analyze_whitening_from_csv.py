#!/usr/bin/env python3
"""
Analyze metrics.csv from 620 experiments for fog/whitening indicators.
Uses existing metrics (clip_style, content_lpips, clip_image_vector) to
quantify whitening without needing actual images.
"""
import csv, json, sys, os
import numpy as np
from pathlib import Path

def parse_clip_vector(s):
    """Parse space-separated clip_image_vector string."""
    try:
        return np.array([float(x) for x in s.strip().split()])
    except:
        return None

def analyze_metrics_csv(csv_path):
    """Analyze a metrics.csv for whitening indicators."""
    rows = list(csv.DictReader(open(csv_path, 'r', encoding='utf-8')))
    if not rows:
        return None

    # Basic metrics
    clip_styles = [float(r['clip_style']) for r in rows if r.get('clip_style')]
    content_lpips = [float(r['content_lpips']) for r in rows if r.get('content_lpips')]
    clip_dirs = [float(r['clip_dir']) for r in rows if r.get('clip_dir')]

    # Parse clip vectors
    vectors = []
    for r in rows:
        v = parse_clip_vector(r.get('clip_image_vector', ''))
        if v is not None:
            vectors.append(v)

    # Split identity vs style_transfer
    idt_styles = [float(r['clip_style']) for r in rows if r.get('clip_style') and r.get('src_style','') == r.get('tgt_style','')]
    st_styles = [float(r['clip_style']) for r in rows if r.get('clip_style') and r.get('src_style','') != r.get('tgt_style','')]

    result = {
        "n_pairs": len(rows),
        "clip_style_mean": float(np.mean(clip_styles)) if clip_styles else None,
        "clip_style_std": float(np.std(clip_styles)) if clip_styles else None,
        "content_lpips_mean": float(np.mean(content_lpips)) if content_lpips else None,
        "content_lpips_std": float(np.std(content_lpips)) if content_lpips else None,
        "clip_dir_mean": float(np.mean(clip_dirs)) if clip_dirs else None,
        "identity_clip_style_mean": float(np.mean(idt_styles)) if idt_styles else None,
        "style_transfer_clip_style_mean": float(np.mean(st_styles)) if st_styles else None,
    }

    # Vector norm analysis (whitening indicator)
    if vectors:
        vecs = np.array(vectors)
        norms = np.linalg.norm(vecs, axis=1)
        result["vector_norm_mean"] = float(np.mean(norms))
        result["vector_norm_std"] = float(np.std(norms))

        # Vector mean direction (if all vectors point same direction = low diversity = whitening)
        mean_vec = np.mean(vecs, axis=0)
        mean_vec_norm = np.linalg.norm(mean_vec)
        result["mean_vector_norm"] = float(mean_vec_norm)

        # Cosine similarity to mean (high = all similar = whitening)
        if mean_vec_norm > 0:
            cos_sims = vecs @ mean_vec / (norms * mean_vec_norm + 1e-8)
            result["cos_sim_to_mean_mean"] = float(np.mean(cos_sims))
            result["cos_sim_to_mean_std"] = float(np.std(cos_sims))

    return result

def main():
    experiments = {
        "620_intrinsic_v2": "/mnt/i/Github/Latent_Style/exp/620_spatial_bridge/620_intrinsic_v2",
        "620_lowswd_formal": "/mnt/i/Github/Latent_Style/exp/620_spatial_bridge/620_lowswd_formal",
        "620_film_formal": "/mnt/i/Github/Latent_Style/exp/620_spatial_bridge/620_film_formal",
    }

    results = {}
    for name, base_dir in experiments.items():
        # Find latest epoch
        full_eval = Path(base_dir) / "full_eval"
        if not full_eval.is_dir():
            print(f"[WARN] {name}: no full_eval dir")
            continue
        epochs = sorted(full_eval.glob("epoch_*"))
        if not epochs:
            print(f"[WARN] {name}: no epochs")
            continue
        latest = epochs[-1]
        csv_path = latest / "metrics.csv"
        if not csv_path.is_file():
            print(f"[WARN] {name}: no metrics.csv in {latest}")
            continue
        print(f"[INFO] {name}: analyzing {csv_path}")
        result = analyze_metrics_csv(str(csv_path))
        if result:
            result["epoch"] = latest.name
            results[name] = result

    # Print comparison
    print("\n" + "="*90)
    print("Whitening Analysis from metrics.csv (clip_image_vector based)")
    print("="*90)
    header = f"{'Experiment':<25s} {'clip_style':>10s} {'cnt_lpips':>10s} {'vec_norm':>10s} {'cos_sim':>10s} {'idt_style':>10s} {'st_style':>10s}"
    print(header)
    print("-"*90)
    for name, r in results.items():
        line = f"{name:<25s} {r.get('clip_style_mean',0):.4f}     {r.get('content_lpips_mean',0):.4f}     {r.get('vector_norm_mean',0):.4f}     {r.get('cos_sim_to_mean_mean',0):.4f}     {r.get('identity_clip_style_mean',0):.4f}     {r.get('style_transfer_clip_style_mean',0):.4f}"
        print(line)

    # Save
    output_path = "/mnt/i/Github/Latent_Style/exp/620_spatial_bridge/whitening_metrics_analysis.json"
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n[INFO] Saved to {output_path}")

if __name__ == "__main__":
    main()
