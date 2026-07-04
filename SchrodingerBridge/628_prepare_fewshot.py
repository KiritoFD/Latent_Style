"""Prepare few-shot datasets for 5+N style experiments.

From existing 5-style base (distinct5), add 1/2/3 new styles with varying shot counts.
New styles from I:\wikiart_latents_512_ema: Expressionism, Post_Impressionism, Realism, Symbolism

Output: packed latent caches + manifests for each (N_new, shots) combination.
Also creates test image dirs for the new styles.
"""
import os
import sys
import json
import shutil
import random
import torch
from pathlib import Path

sys.stdout.reconfigure(encoding='utf-8', errors='replace')

ROOT = Path(r"I:\Github\Latent_Style\SchrodingerBridge")
BASE_PACKED = ROOT / "exp" / "p4_fusion_breakout" / "t5_b2v2_d2_d4"  # has config.json with correct paths
BASE_LATENT_CACHE = Path(r"I:\wikiart_distinct5_samam_512_latents_ema\train\.latent_cache\packed")
NEW_STYLES_LATENTS = Path(r"I:\wikiart_latents_512_ema")
NEW_STYLES_TEST_IMAGES = Path(r"I:\wikiart_images_512_ema_test")
OUTPUT_BASE = Path(r"I:\fewshot_data")

BASE_STYLES = ["Early_Renaissance", "Impressionism", "Minimalism", "Rococo", "Ukiyo_e"]
NEW_STYLE_CANDIDATES = ["Expressionism", "Post_Impressionism", "Realism", "Symbolism"]
SHOT_COUNTS = [1, 6, 10, 30, 50]
SEED = 42


def load_packed_latents(style_name, latent_dir):
    """Load individual latent .pt files for a style and return stacked tensor."""
    style_dir = latent_dir / style_name
    if not style_dir.exists():
        raise FileNotFoundError(f"Latent dir not found: {style_dir}")
    files = sorted([f for f in os.listdir(style_dir) if f.endswith('.pt')])
    latents = []
    for f in files:
        t = torch.load(os.path.join(style_dir, f), map_location='cpu')
        if isinstance(t, dict):
            t = t.get('latent', t.get('latents', list(t.values())[0]))
        if t.dim() == 4 and t.shape[0] == 1:
            t = t.squeeze(0)
        elif t.dim() == 3:
            pass
        latents.append(t)
    return torch.stack(latents), files


def prepare_fewshot_dataset(new_styles, shots, output_dir):
    """Create a packed latent cache for 5+new_styles with given shot count."""
    packed_dir = output_dir / ".latent_cache" / "packed"
    packed_dir.mkdir(parents=True, exist_ok=True)
    
    all_styles = BASE_STYLES + new_styles
    manifest = {
        "schema": 1,
        "data_root": str(output_dir),
        "style_subdirs": all_styles,
        "styles": {},
    }
    
    # Hardlink base styles
    for i, style in enumerate(BASE_STYLES):
        src = BASE_PACKED_DIR / f"{i:02d}_{style}.pt"
        dst = packed_dir / f"{i:02d}_{style}.pt"
        if not src.exists():
            print(f"  WARNING: base packed file not found: {src}")
            continue
        if dst.exists():
            os.remove(dst)
        os.link(str(src), str(dst))
        
        # Read manifest from base
        src_data = torch.load(str(src), map_location='cpu')
        manifest["styles"][style] = {
            "count": src_data.get('count', 1000),
            "files": src_data.get('files', []),
        }
    
    # Create new style packed files
    for j, style in enumerate(new_styles):
        all_latents, all_files = load_packed_latents(style, NEW_STYLES_LATENTS)
        total = len(all_latents)
        
        # Sample `shots` images
        rng = random.Random(SEED)
        indices = sorted(rng.sample(range(total), min(shots, total)))
        selected_latents = all_latents[indices]
        selected_files = [all_files[i] for i in indices]
        
        idx = len(BASE_STYLES) + j
        packed_path = packed_dir / f"{idx:02d}_{style}.pt"
        packed_data = {
            "schema": 1,
            "subdir": style,
            "count": shots,
            "files": selected_files,
            "latents": selected_latents,
        }
        torch.save(packed_data, str(packed_path))
        
        manifest["styles"][style] = {
            "count": shots,
            "files": selected_files,
        }
        print(f"    {style}: {shots}/{total} shots selected, saved to {packed_path.name}")
    
    # Save manifest
    manifest_path = packed_dir / "manifest.json"
    with open(manifest_path, 'w', encoding='utf-8') as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    
    # Create train dir structure (empty dirs for data_root compatibility)
    train_dir = output_dir / "train"
    for style in all_styles:
        (train_dir / style).mkdir(parents=True, exist_ok=True)
    
    # Create test dir with symlinks/copies
    test_dir = output_dir / "test"
    test_dir.mkdir(parents=True, exist_ok=True)
    for style in BASE_STYLES:
        src = Path(r"I:\wikiart_distinct5_samam_512_classview\test") / style
        dst = test_dir / style
        if dst.exists():
            shutil.rmtree(dst)
        shutil.copytree(str(src), str(dst))
    for style in new_styles:
        src = NEW_STYLES_TEST_IMAGES / style
        dst = test_dir / style
        if not src.exists():
            print(f"  WARNING: test images not found for {style}: {src}")
            continue
        if dst.exists():
            shutil.rmtree(dst)
        shutil.copytree(str(src), str(dst))
    
    return output_dir


def main():
    # Find base packed dir
    global BASE_PACKED_DIR
    BASE_PACKED_DIR = BASE_LATENT_CACHE
    if not (BASE_PACKED_DIR / "00_Early_Renaissance.pt").exists():
        # Try the packed/packed subdirectory
        BASE_PACKED_DIR = BASE_LATENT_CACHE / "packed"
    if not (BASE_PACKED_DIR / "00_Early_Renaissance.pt").exists():
        print(f"ERROR: Cannot find base packed latents at {BASE_PACKED_DIR}")
        sys.exit(1)
    print(f"Base packed dir: {BASE_PACKED_DIR}")
    
    # Experiment matrix: (n_new, shots)
    experiments = []
    for n_new in [1, 2, 3]:
        new_styles = NEW_STYLE_CANDIDATES[:n_new]
        for shots in SHOT_COUNTS:
            experiments.append((n_new, new_styles, shots))
    
    print(f"\nPreparing {len(experiments)} few-shot datasets...")
    
    for n_new, new_styles, shots in experiments:
        exp_name = f"5p{n_new}_shot{shots:02d}"
        output_dir = OUTPUT_BASE / exp_name
        print(f"\n=== {exp_name}: +{new_styles}, {shots} shots each ===")
        prepare_fewshot_dataset(new_styles, shots, output_dir)
        print(f"  Output: {output_dir}")
    
    print(f"\nDone! All datasets in {OUTPUT_BASE}")


if __name__ == "__main__":
    main()
