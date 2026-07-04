"""
620 WFI Benchmark - Standalone version for remote execution.
No dependency on local imports. Computes Whitening Fog Index for
620 experiments and Seedream reference images.
"""
from __future__ import annotations
import argparse, csv, json, os, sys
from pathlib import Path
import numpy as np
from PIL import Image

DISTINCT5_STYLES = ["Early_Renaissance", "Impressionism", "Minimalism", "Rococo", "Ukiyo_e"]

def _load_image(path):
    img = Image.open(path).convert("RGB")
    return np.array(img, dtype=np.float32) / 255.0

def compute_image_fog_metrics(img):
    gray = 0.299 * img[:,:,0] + 0.587 * img[:,:,1] + 0.114 * img[:,:,2]
    mean_val = float(np.mean(gray))
    std_val = float(np.std(gray))
    contrast_ratio = std_val / max(mean_val, 1e-8)
    p05, p95 = float(np.percentile(gray, 5)), float(np.percentile(gray, 95))
    dynamic_range = (p95 - p05) / max(p95 + p05, 1e-8)
    r, g, b = img[:,:,0], img[:,:,1], img[:,:,2]
    mx = np.maximum(np.maximum(r, g), b)
    mn = np.minimum(np.minimum(r, g), b)
    delta = mx - mn
    sat = np.where(mx > 0, delta / np.maximum(mx, 1e-8), 0.0)
    saturation_mean = float(np.mean(sat))
    dx, dy = np.diff(gray, axis=1), np.diff(gray, axis=0)
    h2, w2 = min(dx.shape[0], dy.shape[0]), min(dx.shape[1], dy.shape[1])
    edge_energy = float(np.mean(np.sqrt(dx[:h2,:w2]**2 + dy[:h2,:w2]**2)))
    cr_norm = min(contrast_ratio / 0.5, 1.0)
    sr_norm = min(saturation_mean / 0.4, 1.0)
    dr_norm = min(dynamic_range / 0.6, 1.0)
    wfi_score = 1.0 - (0.4 * cr_norm + 0.3 * sr_norm + 0.3 * dr_norm)
    return {
        "contrast_ratio": round(contrast_ratio, 4),
        "dynamic_range": round(dynamic_range, 4),
        "saturation_mean": round(saturation_mean, 4),
        "luminance_std": round(std_val, 4),
        "edge_energy": round(edge_energy, 6),
        "wfi_score": round(wfi_score, 4),
    }

def _summarize(rows):
    if not rows:
        return {"image_count": 0, "failed_count": 0, "metrics": {}}
    valid = [r for r in rows if "error" not in r]
    if not valid:
        return {"image_count": 0, "failed_count": len(rows), "metrics": {}}
    avg = {}
    for k in ["contrast_ratio","dynamic_range","saturation_mean","luminance_std","edge_energy","wfi_score"]:
        vals = [float(r[k]) for r in valid]
        avg[f"avg_{k}"] = round(float(np.mean(vals)), 4)
        avg[f"std_{k}"] = round(float(np.std(vals)), 4)
    return {"image_count": len(valid), "failed_count": len(rows)-len(valid), "metrics": avg}

def parse_style_pair(filename):
    stem = Path(filename).stem
    if "__to__" in stem:
        left, tgt = stem.rsplit("__to__", 1)
        if "__" in left:
            src = left.split("__",1)[0]
            if src in DISTINCT5_STYLES and tgt in DISTINCT5_STYLES:
                return src, tgt
        return None
    if "_to_" not in stem:
        return None
    left, tgt = stem.rsplit("_to_", 1)
    for src in sorted(DISTINCT5_STYLES, key=lambda x: len(x), reverse=True):
        if left.startswith(f"{src}_"):
            if src in DISTINCT5_STYLES and tgt in DISTINCT5_STYLES:
                return src, tgt
    return None

def eval_images(image_paths):
    buckets = {"all": [], "identity": [], "style_transfer": []}
    for p in image_paths:
        try:
            img = _load_image(str(p))
            m = compute_image_fog_metrics(img)
            m["filename"] = p.name
        except Exception as e:
            m = {"filename": p.name, "error": str(e)}
        buckets["all"].append(dict(m))
        pair = parse_style_pair(p.name)
        if pair:
            src, tgt = pair
            if src == tgt:
                buckets["identity"].append(dict(m))
            else:
                buckets["style_transfer"].append(dict(m))
    return {name: _summarize(rows) for name, rows in buckets.items()}

def collect_images(d):
    files = []
    for pat in ("*.png","*.jpg","*.jpeg","*.webp"):
        files.extend(sorted(Path(d).glob(pat)))
    # Also recurse one level into subdirs
    for subdir in sorted(Path(d).iterdir()):
        if subdir.is_dir():
            for pat in ("*.png","*.jpg","*.jpeg","*.webp"):
                files.extend(sorted(subdir.glob(pat)))
    return sorted(set(files))

def find_latest_epoch(ckpt_dir):
    for subdir in ("full_eval", "full_eval_transfer"):
        fe = Path(ckpt_dir) / subdir
        if fe.is_dir():
            epochs = sorted(fe.glob("epoch_*"))
            if epochs:
                return epochs[-1]
    return None

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-dir", nargs="+", required=True)
    parser.add_argument("--epoch", default=None)
    parser.add_argument("--seedream-dir", default=None)
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    exp_results = {}
    for ckpt in args.checkpoint_dir:
        if args.epoch:
            epoch_name = f"epoch_{args.epoch.zfill(4)}"
            for sub in ("full_eval","full_eval_transfer"):
                candidate = Path(ckpt) / sub / epoch_name
                if candidate.is_dir():
                    epoch_dir = candidate
                    break
            else:
                print(f"[WARN] epoch dir not found in {ckpt}")
                continue
        else:
            epoch_dir = find_latest_epoch(ckpt)
        if epoch_dir is None:
            print(f"[WARN] No epoch found in {ckpt}")
            continue
        images_dir = epoch_dir / "images"
        if not images_dir.is_dir():
            print(f"[WARN] No images/ in {epoch_dir}")
            continue
        name = f"{Path(ckpt).name}"
        print(f"[INFO] {name}: {images_dir}")
        paths = collect_images(images_dir)
        print(f"       {len(paths)} images")
        if paths:
            exp_results[name] = eval_images(paths)
            for g, d in exp_results[name].items():
                cnt = d.get("image_count", 0)
                wfi = d.get("metrics",{}).get("avg_wfi_score","N/A")
                print(f"       {g}: n={cnt}, wfi={wfi}")

    seedream_results = {}
    if args.seedream_dir and Path(args.seedream_dir).is_dir():
        print(f"\n[INFO] Seedream: {args.seedream_dir}")
        paths = collect_images(args.seedream_dir)
        print(f"       {len(paths)} images")
        if paths:
            seedream_results = eval_images(paths)
            for g, d in seedream_results.items():
                cnt = d.get("image_count", 0)
                wfi = d.get("metrics",{}).get("avg_wfi_score","N/A")
                print(f"       {g}: n={cnt}, wfi={wfi}")

    # Output
    output_dir = Path(args.output_dir) if args.output_dir else Path(args.checkpoint_dir[0]) / "wfi_benchmark"
    output_dir.mkdir(parents=True, exist_ok=True)

    comparison = {"experiments": exp_results, "seedream": seedream_results}
    json_path = output_dir / "wfi_benchmark_comparison.json"
    json_path.write_text(json.dumps(comparison, indent=2, ensure_ascii=False)+"\n", encoding="utf-8")
    print(f"\n[INFO] Saved to {json_path}")

    # Print summary
    print("\n" + "="*80)
    print("WFI Benchmark Summary (lower wfi_score = less foggy)")
    print("="*80)
    keys = ["avg_contrast_ratio","avg_dynamic_range","avg_saturation_mean","avg_luminance_std","avg_wfi_score"]
    for name, groups in exp_results.items():
        print(f"\n  {name}")
        for g in ("all","identity","style_transfer"):
            m = groups.get(g,{}).get("metrics",{})
            vals = {k: m.get(k, "N/A") for k in keys}
            print(f"    {g:<18s} wfi={vals['avg_wfi_score']}  cr={vals['avg_contrast_ratio']}  dr={vals['avg_dynamic_range']}  sat={vals['avg_saturation_mean']}  lst={vals['avg_luminance_std']}")
    if seedream_results:
        print(f"\n  Seedream45")
        for g in ("all","identity","style_transfer"):
            m = seedream_results.get(g,{}).get("metrics",{})
            vals = {k: m.get(k, "N/A") for k in keys}
            print(f"    {g:<18s} wfi={vals['avg_wfi_score']}  cr={vals['avg_contrast_ratio']}  dr={vals['avg_dynamic_range']}  sat={vals['avg_saturation_mean']}  lst={vals['avg_luminance_std']}")
    print("="*80)

if __name__ == "__main__":
    main()
