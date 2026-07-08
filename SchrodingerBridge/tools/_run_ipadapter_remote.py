"""Run IP-Adapter on Photo2Art-256 and Random5-WikiArt, then evaluate.
To be executed on remote machine via SSH.
"""
import json, os, sys, time, gc
from pathlib import Path

import torch
from PIL import Image
from diffusers import StableDiffusionImg2ImgPipeline

SEED = 42
DEVICE = "cuda"
DTYPE = torch.float16

# ── Dataset definitions ──
# Photo2Art-256: 5 styles (cezanne, Hayao, monet, photo, vangogh), 256x256
# Path: I:/legacy256_overfit50/test  — if not found, probe I: drive
P2A_STYLES = ["cezanne", "Hayao", "monet", "photo", "vangogh"]
P2A_SIZE = 256

# Random5-WikiArt: 5 hold-out styles from wikiarts20, 512x512
# The hold-out 5 styles are different from Distinct5's 5:
# Based on SaMam Random5 eval, we use the first 5 wikiart20 families
# Actually let's scan and pick 5 hold-out styles
R5_SIZE = 512

OUT_ROOT = Path("I:/GitHub/Latent_Style/SchrodingerBridge/exp/baseline_ipadapter")


def find_test_dir(name_hint, search_root):
    """Find test directory by probing common paths."""
    candidates = [
        Path(f"I:/legacy256_overfit50/test"),
        Path(f"I:/datasets/{name_hint}"),
        Path(f"I:/datasets/wikiarts20_512_test"),
        Path(f"I:/wikiart_distinct5_samam_512_classview/test"),
        Path(f"I:/datasets"),
    ]
    for p in candidates:
        if p.exists():
            print(f"  Found: {p}")
            return p
    return None


def scan_styles(test_dir, expected_styles=None):
    """Scan a test directory for style subdirectories."""
    if not test_dir or not test_dir.exists():
        return []
    styles = sorted([d.name for d in test_dir.iterdir() if d.is_dir() and not d.name.startswith('.')])
    if expected_styles:
        styles = [s for s in expected_styles if (test_dir / s).exists()]
    return styles


def load_ip_adapter_pipe():
    """Load SD1.5 img2img pipeline with IP-Adapter."""
    print("  Loading SD1.5 img2img + IP-Adapter ...", flush=True)
    pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        torch_dtype=DTYPE,
        safety_checker=None,
        requires_safety_checker=False,
    )
    pipe = pipe.to(DEVICE)
    pipe.load_ip_adapter("h94/IP-Adapter", subfolder="models", weight_name="ip-adapter-plus_sd15.safetensors")
    pipe.set_ip_adapter_scale(0.7)
    return pipe


def run_inference(pipe, test_dir, style_list, out_dir, img_size, dataset_label):
    """Run IP-Adapter inference on all style pairs."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Collect all source images with (style, path) pairs
    sources = []
    for style in style_list:
        style_dir = test_dir / style
        if not style_dir.exists():
            print(f"  WARNING: {style_dir} not found, skipping style {style}")
            continue
        for f in sorted(style_dir.iterdir()):
            if f.suffix.lower() in {".jpg", ".jpeg", ".png"}:
                sources.append((style, f))
    
    # Pick style references (first image per target style)
    style_refs = {}
    for tgt_style in style_list:
        style_dir = test_dir / tgt_style
        if style_dir.exists():
            refs = sorted([f for f in style_dir.iterdir() if f.suffix.lower() in {".jpg", ".jpeg", ".png"}])
            if refs:
                style_refs[tgt_style] = refs[0]
    
    total_pairs = len(sources) * len(style_list)
    print(f"  [{dataset_label}] {len(sources)} src × {len(style_list)} tgt = {total_pairs} pairs", flush=True)
    
    generator = torch.Generator(device=DEVICE).manual_seed(SEED)
    count = 0
    start_all = time.time()
    
    for src_style, src_path in sources:
        src_stem = src_path.stem
        # Remove __prefix if any
        if "__" in src_stem:
            src_stem = src_stem.split("__", 1)[1]
        
        src_img = Image.open(src_path).convert("RGB")
        if src_img.size != (img_size, img_size):
            src_img = src_img.resize((img_size, img_size), Image.LANCZOS)
        
        for tgt_style in style_list:
            out_name = f"{src_style}__{src_stem}__to__{tgt_style}.png"
            out_path = out_dir / out_name
            if out_path.exists():
                count += 1
                continue
            
            if tgt_style not in style_refs:
                continue
            
            ref_img = Image.open(style_refs[tgt_style]).convert("RGB")
            if ref_img.size != (img_size, img_size):
                ref_img = ref_img.resize((img_size, img_size), Image.LANCZOS)
            
            with torch.no_grad():
                result = pipe(
                    prompt="",
                    image=src_img,
                    ip_adapter_image=ref_img,
                    strength=0.65,
                    num_inference_steps=20,
                    guidance_scale=7.5,
                    generator=generator,
                ).images[0]
            
            result.save(str(out_path))
            count += 1
        
        if count % 50 == 0:
            elapsed = time.time() - start_all
            print(f"  [{dataset_label}] {count}/{total_pairs}  elapsed={elapsed/60:.1f}m", flush=True)
    
    total_time = time.time() - start_all
    print(f"  [{dataset_label}] Done. {total_time/60:.1f} min total", flush=True)
    
    # Save metadata
    meta = {
        "method": "ip_adapter_plus_sd15_img2img",
        "dataset": dataset_label,
        "test_dir": str(test_dir),
        "out_dir": str(out_dir),
        "img_size": img_size,
        "style_list": style_list,
        "total_pairs": total_pairs,
        "total_generated": count,
        "total_seconds": total_time,
    }
    meta_path = out_dir / "metadata.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(f"  Metadata saved to {meta_path}", flush=True)


def main():
    print("=" * 60, flush=True)
    print("IP-Adapter Remote Inference — Photo2Art-256 + Random5-WikiArt", flush=True)
    print("=" * 60, flush=True)
    
    # ── Photo2Art-256 ──
    print("\n[1/2] Photo2Art-256", flush=True)
    p2a_test = find_test_dir("legacy256_overfit50/test", "I:")
    if p2a_test and p2a_test.exists():
        styles = scan_styles(p2a_test, P2A_STYLES)
        print(f"  Styles: {styles}", flush=True)
        pipe = load_ip_adapter_pipe()
        out = OUT_ROOT / "photo2art256" / "images"
        run_inference(pipe, p2a_test, styles, out, P2A_SIZE, "P2A")
        del pipe; gc.collect(); torch.cuda.empty_cache(); time.sleep(2)
    else:
        print(f"  SKIP: Photo2Art-256 test dir not found at {p2a_test}", flush=True)
    
    # ── Random5-WikiArt ──
    print("\n[2/2] Random5-WikiArt", flush=True)
    r5_test = Path("I:/datasets/wikiarts20_512_test")
    # Random5 uses the first 5 styles as hold-out (different from Distinct5's 5)
    # wikiarts20 has these families:
    # Abstract_Expressionism, Analytic_Cubism, Baroque, Color_Field_Painting, Contemporary_Realism,
    # Cubism, Early_Renaissance, Expressionism, Fauvism, High_Renaissance, Impressionism,
    # Mannerism_Late_Renaissance, Minimalism, Naive_Art_Primitivism, Northern_Renaissance,
    # Pointillism, Pop_Art, Post_Impressionism, Realism, Rococo, Romanticism, 
    # Symbolism, Synthetic_Cubism, Ukiyo_e
    # Distinct5 = {Early_Renaissance, Impressionism, Minimalism, Rococo, Ukiyo_e}
    # Random5 hold-out = pick 5 others, e.g.:
    R5_HOLDOUT = ["Cubism", "Expressionism", "Pop_Art", "Pointillism", "Romanticism"]
    
    if r5_test.exists():
        styles = scan_styles(r5_test, R5_HOLDOUT)
        if len(styles) < 5:
            # Fallback: use the first 5 alphabetically that aren't in Distinct5
            all_styles = scan_styles(r5_test)
            distinct5 = {"Early_Renaissance", "Impressionism", "Minimalism", "Rococo", "Ukiyo_e"}
            other = sorted([s for s in all_styles if s not in distinct5])
            styles = other[:5]
        print(f"  Styles: {styles}", flush=True)
        pipe = load_ip_adapter_pipe()
        out = OUT_ROOT / "random5" / "images"
        run_inference(pipe, r5_test, styles, out, R5_SIZE, "R5")
        del pipe; gc.collect(); torch.cuda.empty_cache()
    else:
        print(f"  SKIP: Random5 test dir not found at {r5_test}", flush=True)
    
    print("\n" + "=" * 60, flush=True)
    print("All done!", flush=True)


if __name__ == "__main__":
    main()
