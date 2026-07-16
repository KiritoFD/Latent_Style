"""Exp B1: IDT vs WEAVE DINO-S margin sensitivity to reference pool.

Following reviewer_audit Section 5 (B1).
For each target style's reference pool (30 imgs), do fixed-size WITHOUT-replacement
subsets (m=8, m=16), 1000 iterations each. Record:
  DINO-S_WEAVE (max cosine gen, ref_subset)
  DINO-S_IDT   (max cosine source, ref_subset)  -- IDT = y=x
  Delta_style = DINO-S_WEAVE - DINO-S_IDT
Report median, 95% CI, min of margin.

NO image regeneration needed; reuses cached DINOv2 CLS features.
"""
import argparse, csv, json, os, sys, time
from pathlib import Path
import numpy as np

WEAVE_ROOT = Path(r"I:\Github\Latent_Style\WEAVE")
os.chdir(WEAVE_ROOT)

EVAL_DIR = WEAVE_ROOT / "exp" / "repro_weave_d5"
TEST_DIR = WEAVE_ROOT / "data" / "test"
HF_CACHE = "exp/eval_cache/hf"
OUTPUT_DIR = WEAVE_ROOT / "exp" / "rebuttal" / "expB_reference_sensitivity_corrected"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def load_dino_features():
    """Load DINOv2-small and extract CLS features for all images."""
    import torch
    import torch.nn.functional as F
    import torchvision.transforms as T
    from PIL import Image
    from transformers import AutoModel

    DINO_TRANSFORM = T.Compose([
        T.Resize(224, interpolation=Image.BICUBIC),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"

    cache_dir = HF_CACHE
    model_name = "facebook/dinov2-small"
    parts = model_name.split("/")
    repo_dir = Path(cache_dir) / "hub" / f"models--{parts[0]}--{parts[1]}"
    snap_root = repo_dir / "snapshots"
    if snap_root.exists():
        revisions = [p for p in snap_root.iterdir() if p.is_dir()]
        if revisions:
            local_path = str(revisions[0])
            print(f"Loading DINOv2 from: {local_path}")
            model = AutoModel.from_pretrained(local_path).to("cuda").eval()
        else:
            model = AutoModel.from_pretrained(model_name, cache_dir=cache_dir).to("cuda").eval()
    else:
        model = AutoModel.from_pretrained(model_name, cache_dir=cache_dir).to("cuda").eval()

    @torch.inference_mode()
    def extract(paths, batch_size=8):
        feats = []
        for start in range(0, len(paths), batch_size):
            batch = paths[start:start + batch_size]
            pixels = torch.stack([DINO_TRANSFORM(Image.open(p).convert("RGB")) for p in batch]).to("cuda")
            out = model(pixels, output_hidden_states=True)
            cls = F.normalize(out.last_hidden_state[:, 0, :].float(), dim=-1).cpu()
            feats.append(cls)
        return torch.cat(feats, dim=0)

    return extract


def main():
    parser = argparse_argument_parser()
    args = parser.parse_args()

    print("=" * 60)
    print("Exp B1: Reference-pool paired margin sensitivity")
    print("=" * 60)

    # Read metrics.csv for generated image paths
    metrics_csv = EVAL_DIR / "metrics.csv"
    if not metrics_csv.exists():
        print(f"ERROR: {metrics_csv} not found")
        sys.exit(1)

    rows = list(csv.DictReader(metrics_csv.open(encoding="utf-8-sig")))
    print(f"Total rows: {len(rows)}")

    # Focus on off-diagonal (src_style != tgt_style) to avoid self-match issues
    off_rows = [r for r in rows if r["src_style"] != r["tgt_style"]]
    print(f"Off-diagonal rows: {len(off_rows)}")

    # Collect all image paths
    extract = load_dino_features()

    # Generated images
    gen_paths = [EVAL_DIR / "images" / r["gen_image"] for r in off_rows]
    # Source images
    src_paths = [TEST_DIR / r["src_style"] / r["src_image"] for r in off_rows]
    # Fix path if needed
    for i, p in enumerate(src_paths):
        if not p.exists():
            alt = TEST_DIR / r["src_style"] / f"{off_rows[i]['src_style']}__{off_rows[i]['src_image']}"
            if alt.exists():
                src_paths[i] = alt

    # Reference images per style
    ref_paths_by_style = {}
    for style_dir in sorted(TEST_DIR.iterdir()):
        if style_dir.is_dir():
            refs = sorted([p for p in style_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS])
            ref_paths_by_style[style_dir.name] = refs
    print(f"Style families: {list(ref_paths_by_style.keys())}")
    for s, refs in ref_paths_by_style.items():
        print(f"  {s}: {len(refs)} refs")

    # Extract features
    print("\nExtracting DINOv2 features...")
    t0 = time.time()
    gen_feats = extract(gen_paths)
    print(f"  Generated: {gen_feats.shape} ({time.time()-t0:.1f}s)")

    t0 = time.time()
    unique_src = list(dict.fromkeys(src_paths))
    src_feats_all = extract(unique_src)
    src_idx = {p: i for i, p in enumerate(unique_src)}
    src_feats = src_feats_all[[src_idx[p] for p in src_paths]]
    print(f"  Source: {src_feats.shape} ({time.time()-t0:.1f}s)")

    t0 = time.time()
    ref_feats_by_style = {}
    for style, refs in ref_paths_by_style.items():
        ref_feats_by_style[style] = extract(refs)
    print(f"  References: ({time.time()-t0:.1f}s)")

    # Convert to numpy for fast sampling
    gen_np = gen_feats.numpy()  # (N, D)
    src_np = src_feats.numpy()  # (N, D)
    ref_np = {s: f.numpy() for s, f in ref_feats_by_style.items()}  # {style: (R, D)}

    # For each row, get target style
    tgt_styles = [r["tgt_style"] for r in off_rows]

    # Each draw defines one reference pool per target style. Reusing that pool
    # for every request to the style measures pool sensitivity rather than
    # per-image reference noise.
    rows_by_target = {}
    for index, target_style in enumerate(tgt_styles):
        rows_by_target.setdefault(target_style, []).append(index)
    rng = np.random.default_rng(20260716)

    # B1: Subset sampling
    results = {}
    for pool_size in [8, 16]:
        print(f"\n--- Pool size m={pool_size} ---")
        n_iters = args.n_iters
        margins = np.zeros((n_iters, len(off_rows)))
        weave_scores = np.zeros((n_iters, len(off_rows)))
        idt_scores = np.zeros((n_iters, len(off_rows)))

        for it in range(n_iters):
            if it % 100 == 0:
                print(f"  Iter {it}/{n_iters}...", flush=True)
            for tgt, indices in rows_by_target.items():
                full_pool = ref_np[tgt]
                R = full_pool.shape[0]
                if R <= pool_size:
                    subset = full_pool
                else:
                    idx = rng.choice(R, size=pool_size, replace=False)
                    subset = full_pool[idx]
                selected = np.asarray(indices)
                weave_scores[it, selected] = (gen_np[selected] @ subset.T).max(axis=1)
                idt_scores[it, selected] = (src_np[selected] @ subset.T).max(axis=1)
            margins[it] = weave_scores[it] - idt_scores[it]

        board_margin = margins.mean(axis=1)
        board_weave = weave_scores.mean(axis=1)
        board_idt = idt_scores.mean(axis=1)

        results[f"m{pool_size}"] = {
            "weave_dino_s": {
                "mean": float(board_weave.mean()),
                "std": float(board_weave.std()),
                "median": float(np.median(board_weave)),
                "ci_lower": float(np.percentile(board_weave, 2.5)),
                "ci_upper": float(np.percentile(board_weave, 97.5)),
            },
            "idt_dino_s": {
                "mean": float(board_idt.mean()),
                "std": float(board_idt.std()),
                "median": float(np.median(board_idt)),
                "ci_lower": float(np.percentile(board_idt, 2.5)),
                "ci_upper": float(np.percentile(board_idt, 97.5)),
            },
            "margin": {
                "mean": float(board_margin.mean()),
                "std": float(board_margin.std()),
                "median": float(np.median(board_margin)),
                "ci_lower": float(np.percentile(board_margin, 2.5)),
                "ci_upper": float(np.percentile(board_margin, 97.5)),
                "min": float(board_margin.min()),
                "frac_positive_board": float((board_margin > 0).mean()),
                "frac_positive_request": float((margins > 0).mean()),
            },
        }

        print(f"  WEAVE DINO-S: {results[f'm{pool_size}']['weave_dino_s']['mean']:.4f} ± {results[f'm{pool_size}']['weave_dino_s']['std']:.4f}")
        print(f"  IDT DINO-S:   {results[f'm{pool_size}']['idt_dino_s']['mean']:.4f} ± {results[f'm{pool_size}']['idt_dino_s']['std']:.4f}")
        print(f"  Margin:       {results[f'm{pool_size}']['margin']['mean']:.4f} ± {results[f'm{pool_size}']['margin']['std']:.4f}")
        print(f"  Positive board margins: {results[f'm{pool_size}']['margin']['frac_positive_board']*100:.1f}%")
        print(f"  Margin 95% CI: [{results[f'm{pool_size}']['margin']['ci_lower']:.4f}, {results[f'm{pool_size}']['margin']['ci_upper']:.4f}]")

    # Full pool (m=30) as baseline
    print("\n--- Full pool m=30 (baseline) ---")
    weave_full = np.array([(ref_np[tgt] @ gen_np[j]).max() for j, tgt in enumerate(tgt_styles)])
    idt_full = np.array([(ref_np[tgt] @ src_np[j]).max() for j, tgt in enumerate(tgt_styles)])
    margin_full = weave_full - idt_full
    results["m30"] = {
        "weave_dino_s": {"mean": float(weave_full.mean()), "std": float(weave_full.std())},
        "idt_dino_s": {"mean": float(idt_full.mean()), "std": float(idt_full.std())},
        "margin": {
            "mean": float(margin_full.mean()),
            "std": float(margin_full.std()),
            "min": float(margin_full.min()),
            "frac_positive": float((margin_full > 0).mean()),
        },
    }
    print(f"  WEAVE DINO-S: {results['m30']['weave_dino_s']['mean']:.4f}")
    print(f"  IDT DINO-S:   {results['m30']['idt_dino_s']['mean']:.4f}")
    print(f"  Margin:       {results['m30']['margin']['mean']:.4f}")
    print(f"  Margin > 0:   {results['m30']['margin']['frac_positive']*100:.1f}%")

    # Save
    out_path = OUTPUT_DIR / "dino_margin.json"
    out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nSaved: {out_path}")

    # Per-row CSV
    csv_path = OUTPUT_DIR / "dino_margin_per_row.csv"
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["src_style", "tgt_style", "src_image", "weave_dino_s_full", "idt_dino_s_full", "margin_full"])
        for j, r in enumerate(off_rows):
            w.writerow([r["src_style"], r["tgt_style"], r["src_image"], weave_full[j], idt_full[j], margin_full[j]])
    print(f"Per-row CSV: {csv_path}")
    print("EXPB1_EXIT=0")


def argparse_argument_parser():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--n_iters", type=int, default=1000)
    return p


if __name__ == "__main__":
    main()
