"""Scan all dataset directories on I drive root, output structured registry."""
import os
import json
from pathlib import Path

ROOT = Path("/mnt/i")

# Candidate dataset directories (exclude code/exp/workspace dirs)
DATASET_DIRS = [
    "Scitexture_latent_512_smoke_ema",
    "datasets",
    "fewshot_data",
    "legacy256_overfit50",
    "legacy256_overfit50_latent256",
    "legacy256_overfit50_pixel256",
    "wikiart_distinct5_latents_512_ema",
    "wikiart_distinct5_latents_512_ema_test",
    "wikiart_distinct5_samam_512_classview",
    "wikiart_distinct5_samam_512_classview_real",
    "wikiart_distinct5_samam_512_flat",
    "wikiart_distinct5_samam_512_latent256",
    "wikiart_distinct5_samam_512_latents_ema",
    "wikiart_distinct5_samam_512_pixel128",
    "wikiart_distinct5_samam_512_pixel256",
    "wikiart_faraday_splits",
    "wikiart_images_512_ema_test",
    "wikiart_latents_512_ema",
    "wikiart_latents_512_ema_test",
    "wikiarts_5_full_notest",
    "wikiarts_5_full_notest_latents_ema",
    # exp dirs (not datasets, but may contain useful artifacts)
    "exp_256_photo2art",
    "exp_our_models_eval",
    "exp_samam_latent",
    "exp_samst_latent",
    "exp_samst_latent_eval",
]


def dir_size_mb(path: Path) -> float:
    total = 0
    try:
        for root, _, files in os.walk(path):
            for f in files:
                try:
                    total += os.path.getsize(os.path.join(root, f))
                except OSError:
                    pass
    except OSError:
        pass
    return total / (1024 * 1024)


def count_files(path: Path, ext_filter=None) -> int:
    n = 0
    try:
        for root, _, files in os.walk(path):
            for f in files:
                if ext_filter is None or any(f.endswith(e) for e in ext_filter):
                    n += 1
    except OSError:
        pass
    return n


def detect_dataset_kind(path: Path) -> dict:
    """Inspect top-level structure to classify the dataset."""
    info = {"subdirs": [], "sample_files": [], "has_train": False, "has_test": False,
            "has_latent_cache": False, "has_manifest": False, "kind": "unknown"}
    try:
        entries = sorted(os.listdir(path))
    except OSError:
        return info
    subdirs = []
    sample_files = []
    for e in entries:
        full = path / e
        if full.is_dir():
            subdirs.append(e)
            if e.lower() == "train":
                info["has_train"] = True
            elif e.lower() == "test":
                info["has_test"] = True
            elif e == ".latent_cache":
                info["has_latent_cache"] = True
        else:
            sample_files.append(e)
            if "manifest" in e.lower() or e.endswith(".json"):
                info["has_manifest"] = True
    info["subdirs"] = subdirs[:30]
    info["sample_files"] = sample_files[:10]
    # Classify kind
    name = path.name.lower()
    if "latent" in name and ("ema" in name or "256" in name or "512" in name):
        info["kind"] = "latent"
    elif "pixel" in name:
        info["kind"] = "pixel"
    elif "classview" in name:
        info["kind"] = "classview_test"
    elif "flat" in name:
        info["kind"] = "flat_pixel"
    elif "scitexture" in name:
        info["kind"] = "scitexture"
    elif "overfit" in name:
        info["kind"] = "overfit"
    elif "fewshot" in name:
        info["kind"] = "fewshot"
    elif "faraday" in name:
        info["kind"] = "splits"
    elif name.startswith("exp_"):
        info["kind"] = "exp_artifacts"
    elif name == "datasets":
        info["kind"] = "container"
    return info


def main():
    print(f"=== I drive dataset inventory ===")
    print(f"Root: {ROOT}\n")
    results = []
    for d in DATASET_DIRS:
        path = ROOT / d
        if not path.exists():
            print(f"[MISSING] {d}")
            continue
        size_mb = dir_size_mb(path)
        n_files = count_files(path)
        info = detect_dataset_kind(path)
        results.append({
            "name": d,
            "path": str(path),
            "size_mb": round(size_mb, 1),
            "size_gb": round(size_mb / 1024, 3),
            "n_files": n_files,
            **info,
        })
        print(f"[{info['kind']:14s}] {d:50s}  {size_mb:>10.1f} MB  {n_files:>7d} files")
        if info["subdirs"]:
            print(f"               subdirs: {info['subdirs'][:10]}")
        if info["sample_files"]:
            print(f"               sample_files: {info['sample_files'][:5]}")

    # Save JSON
    out = Path("/mnt/i/_dataset_registry.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n=== Saved registry to {out} ===")
    print(f"Total datasets scanned: {len(results)}")
    total_gb = sum(r["size_mb"] for r in results) / 1024
    print(f"Total size: {total_gb:.2f} GB")


if __name__ == "__main__":
    main()
