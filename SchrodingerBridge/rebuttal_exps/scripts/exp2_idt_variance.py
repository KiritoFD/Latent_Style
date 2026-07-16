"""Exp2: IDT-TGT Sandwich variance analysis.
Uses existing generated images at exp/repro_weave_d5/.
Randomly samples 30 different reference subsets, computes DINO-S and IDT floor
for each, reports mean/std/CI.

Run on remote RTX 3060.
"""
import json, os, sys, random, time, csv
from pathlib import Path
import torch
import torch.nn.functional as F
from PIL import Image
import torchvision.transforms as T

WEAVE_ROOT = Path(r"I:\Github\Latent_Style\WEAVE")
os.chdir(WEAVE_ROOT)

GEN_DIR = WEAVE_ROOT / "exp" / "repro_weave_d5" / "images"
TEST_DIR = WEAVE_ROOT / "data" / "test"
METRICS_CSV = WEAVE_ROOT / "exp" / "repro_weave_d5" / "metrics.csv"
OUTPUT = WEAVE_ROOT / "exp" / "rebuttal" / "exp2_idt_variance.json"
FEATS_CACHE = WEAVE_ROOT / "exp" / "rebuttal" / "exp2_features.pt"
N_ITERS = 30
N_REFS_PER_STYLE = 30  # Sample with replacement (bootstrap) from available refs
N_REFS_SUBSAMPLE = 10  # Also test with smaller subsets without replacement
DEVICE = "cuda"

DINO_TRANSFORM = T.Compose([
    T.Resize(224, interpolation=Image.BICUBIC),
    T.CenterCrop(224),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def load_dino_model():
    """Load DINOv2-small model from local cache (offline)."""
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    from transformers import AutoModel
    cache_dir = str(WEAVE_ROOT / "exp" / "eval_cache" / "hf")
    # Try local snapshot first
    repo_dir = Path(cache_dir) / "hub" / "models--facebook--dinov2-small"
    snap_root = repo_dir / "snapshots"
    if snap_root.exists():
        revisions = [p for p in snap_root.iterdir() if p.is_dir()]
        if revisions:
            local_path = str(revisions[0])
            print(f"Loading DINOv2 from local snapshot: {local_path}")
            model = AutoModel.from_pretrained(local_path).to(DEVICE).eval()
            return model
    # Fallback to cache_dir
    model = AutoModel.from_pretrained("facebook/dinov2-small", cache_dir=cache_dir).to(DEVICE).eval()
    return model

def get_dino_features(model, images):
    """Get normalized CLS features for a list of PIL images."""
    pixels = torch.stack([DINO_TRANSFORM(img) for img in images]).to(DEVICE)
    with torch.inference_mode():
        outputs = model(pixels)
        cls = outputs.last_hidden_state[:, 0, :].float()
    return F.normalize(cls, dim=-1)

def collect_reference_images():
    """Collect all reference images per style from test_dir."""
    refs_by_style = {}
    for style_dir in sorted(TEST_DIR.iterdir()):
        if not style_dir.is_dir():
            continue
        imgs = sorted(list(style_dir.glob("*.png")) + list(style_dir.glob("*.jpg")))
        refs_by_style[style_dir.name] = imgs
    return refs_by_style

def collect_generated_images():
    """Collect generated images from metrics.csv, grouped by target style."""
    gen_by_style = {}
    if not METRICS_CSV.exists():
        print(f"WARNING: metrics.csv not found at {METRICS_CSV}")
        return gen_by_style
    with open(METRICS_CSV, encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for r in reader:
            tgt_style = r.get("tgt_style", r.get("target_style", ""))
            src_style = r.get("src_style", r.get("source_style", ""))
            gen_name = r.get("gen_image", "")
            gen_path = GEN_DIR / gen_name if gen_name else None
            if gen_path and gen_path.exists() and tgt_style:
                if tgt_style not in gen_by_style:
                    gen_by_style[tgt_style] = []
                gen_by_style[tgt_style].append({
                    "path": gen_path,
                    "src_style": src_style,
                    "tgt_style": tgt_style,
                    "name": gen_name,
                })
    return gen_by_style

def compute_dino_s(gen_feats_by_style, ref_feats_by_style):
    """Compute DINO-S = max cosine CLS(gen), CLS(target-style reference)."""
    all_sims = []
    for tgt_style, gen_list in gen_feats_by_style.items():
        if tgt_style not in ref_feats_by_style:
            continue
        ref_feats = ref_feats_by_style[tgt_style]  # (N_ref, D)
        for gen_feat in gen_list:
            sims = (gen_feat.unsqueeze(0) @ ref_feats.T).squeeze(0)
            all_sims.append(sims.max().item())
    return sum(all_sims) / len(all_sims) if all_sims else 0.0

def load_metrics_csv():
    """Load metrics.csv and extract identity pairs."""
    if not METRICS_CSV.exists():
        return []
    rows = []
    with open(METRICS_CSV, encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)
    idt_pairs = []
    for r in rows:
        src_s = r.get("src_style", r.get("source_style", ""))
        tgt_s = r.get("tgt_style", r.get("target_style", ""))
        if src_s == tgt_s:
            clip_s = float(r.get("clip_style", 0))
            idt_pairs.append(clip_s)
    return idt_pairs

def main():
    print("=" * 60)
    print("Exp2: IDT-TGT Sandwich Variance Analysis")
    print("=" * 60)

    # Load DINO model
    print("Loading DINOv2-small...", flush=True)
    model = load_dino_model()

    # Collect images
    print("Collecting reference images...", flush=True)
    refs_by_style = collect_reference_images()
    print(f"  Styles: {list(refs_by_style.keys())}")
    for s, imgs in refs_by_style.items():
        print(f"  {s}: {len(imgs)} refs")

    print("Collecting generated images...", flush=True)
    gen_by_style = collect_generated_images()
    for s, gen_list in gen_by_style.items():
        print(f"  {s}: {len(gen_list)} generated")

    # Pre-compute or load cached features
    if FEATS_CACHE.exists():
        print(f"Loading cached features from {FEATS_CACHE}...", flush=True)
        cache = torch.load(FEATS_CACHE, map_location="cpu")
        gen_feats_by_style = cache["gen_feats"]
        all_ref_feats = cache["ref_feats"]
        print(f"  Loaded: {sum(len(v) for v in gen_feats_by_style.values())} gen, {sum(v.shape[0] for v in all_ref_feats.values())} refs")
    else:
        # Pre-compute generated image features (fixed across iterations)
        print("Computing generated image features...", flush=True)
        gen_feats_by_style = {}
        for tgt_style, gen_list in gen_by_style.items():
            feats = []
            for item in gen_list:
                img = Image.open(item["path"]).convert("RGB")
                feat = get_dino_features(model, [img])
                feats.append(feat.squeeze(0))
            gen_feats_by_style[tgt_style] = feats
            print(f"  {tgt_style}: {len(feats)} features computed")

        # Pre-compute features for ALL reference images
        print("Computing all reference features...", flush=True)
        all_ref_feats = {}
        for style, ref_paths in refs_by_style.items():
            imgs = [Image.open(p).convert("RGB") for p in ref_paths]
            feats = get_dino_features(model, imgs)
            all_ref_feats[style] = feats
            print(f"  {style}: {feats.shape}")

        # Save cache
        FEATS_CACHE.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"gen_feats": gen_feats_by_style, "ref_feats": all_ref_feats}, FEATS_CACHE)
        print(f"  Cached features to {FEATS_CACHE}")

    # Bootstrap with replacement (N_REFS_PER_STYLE from available refs)
    print(f"\nRunning {N_ITERS} bootstrap iterations (N_REFS={N_REFS_PER_STYLE}, with replacement)...", flush=True)
    dino_s_bootstrap = []
    rng = random.Random(42)
    for i in range(N_ITERS):
        ref_feats_subset = {}
        for style, feats in all_ref_feats.items():
            # Bootstrap: sample WITH replacement
            indices = rng.choices(range(feats.shape[0]), k=N_REFS_PER_STYLE)
            ref_feats_subset[style] = feats[indices]
        dino_s = compute_dino_s(gen_feats_by_style, ref_feats_subset)
        dino_s_bootstrap.append(dino_s)
        print(f"  Iter {i+1}/{N_ITERS}: DINO-S={dino_s:.6f}", flush=True)

    # Subsample without replacement (N_REFS_SUBSAMPLE from available refs)
    print(f"\nRunning {N_ITERS} subsample iterations (N_REFS={N_REFS_SUBSAMPLE}, without replacement)...", flush=True)
    dino_s_subsample = []
    rng2 = random.Random(42)
    for i in range(N_ITERS):
        ref_feats_subset = {}
        for style, feats in all_ref_feats.items():
            n = min(N_REFS_SUBSAMPLE, feats.shape[0])
            indices = rng2.sample(range(feats.shape[0]), n)
            ref_feats_subset[style] = feats[indices]
        dino_s = compute_dino_s(gen_feats_by_style, ref_feats_subset)
        dino_s_subsample.append(dino_s)
        print(f"  Iter {i+1}/{N_ITERS}: DINO-S={dino_s:.6f}", flush=True)

    # IDT variance via bootstrap
    print("\nComputing IDT variance via bootstrap...", flush=True)
    idt_pairs = load_metrics_csv()
    print(f"  Identity pairs: {len(idt_pairs)}")
    if idt_pairs:
        idt_results = []
        for i in range(N_ITERS):
            sample = rng.choices(idt_pairs, k=len(idt_pairs))
            idt_results.append(sum(sample) / len(sample))
        idt_mean = sum(idt_results) / len(idt_results)
        idt_std = (sum((x - idt_mean) ** 2 for x in idt_results) / len(idt_results)) ** 0.5
        idt_ci = sorted(idt_results)[int(0.025 * N_ITERS)], sorted(idt_results)[int(0.975 * N_ITERS)]
    else:
        idt_mean = idt_std = 0
        idt_ci = (0, 0)

    # Summary
    def summarize(values):
        mean = sum(values) / len(values)
        std = (sum((x - mean) ** 2 for x in values) / len(values)) ** 0.5
        sorted_v = sorted(values)
        ci_low = sorted_v[int(0.025 * len(values))]
        ci_high = sorted_v[int(0.975 * len(values)) - 1] if len(values) > 1 else sorted_v[0]
        return {"mean": mean, "std": std, "ci_95": [ci_low, ci_high], "all_values": values}

    boot_summary = summarize(dino_s_bootstrap)
    subsample_summary = summarize(dino_s_subsample)

    result = {
        "n_iters": N_ITERS,
        "n_refs_per_style": N_REFS_PER_STYLE,
        "n_refs_subsample": N_REFS_SUBSAMPLE,
        "dino_s_bootstrap": boot_summary,
        "dino_s_subsample": subsample_summary,
        "idt_floor": {
            "mean": idt_mean,
            "std": idt_std,
            "ci_95": [idt_ci[0], idt_ci[1]],
            "n_pairs": len(idt_pairs),
        },
    }

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(result, indent=2))
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"DINO-S bootstrap (N={N_REFS_PER_STYLE}, with repl): {boot_summary['mean']:.6f} ± {boot_summary['std']:.6f} (95% CI: [{boot_summary['ci_95'][0]:.6f}, {boot_summary['ci_95'][1]:.6f}])")
    print(f"DINO-S subsample (N={N_REFS_SUBSAMPLE}, wo repl): {subsample_summary['mean']:.6f} ± {subsample_summary['std']:.6f} (95% CI: [{subsample_summary['ci_95'][0]:.6f}, {subsample_summary['ci_95'][1]:.6f}])")
    print(f"IDT floor: {idt_mean:.6f} ± {idt_std:.6f} (95% CI: [{idt_ci[0]:.6f}, {idt_ci[1]:.6f}])")
    print(f"\nSaved to: {OUTPUT}")

if __name__ == "__main__":
    main()
