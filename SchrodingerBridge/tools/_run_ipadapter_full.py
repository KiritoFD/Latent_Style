"""Run IP-Adapter on Photo2Art-256 and Random5-WikiArt, then evaluate.
Execute on remote: python C:\temp\_run_ipadapter_full.py
"""
import json, os, sys, time, gc, subprocess
from pathlib import Path
import torch
from PIL import Image
from diffusers import StableDiffusionImg2ImgPipeline

SEED = 42
DEVICE = "cuda"
DTYPE = torch.float16
SCALE = 0.7
STRENGTH = 0.65
STEPS = 20
GUIDANCE = 7.5
PROMPT = ""

OUT_ROOT = Path(r"I:\GitHub\Latent_Style\SchrodingerBridge\exp\baseline_ipadapter")
EVAL_SCRIPT = Path(r"I:\GitHub\Latent_Style\SchrodingerBridge\src\utils\run_evaluation.py")


def load_pipe():
    print("  Loading SD1.5 img2img + IP-Adapter-Plus ...", flush=True)
    pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        torch_dtype=DTYPE, safety_checker=None, requires_safety_checker=False,
    )
    pipe = pipe.to(DEVICE)
    pipe.load_ip_adapter("h94/IP-Adapter", subfolder="models", weight_name="ip-adapter-plus_sd15.safetensors")
    pipe.set_ip_adapter_scale(SCALE)
    return pipe


def run_inference(pipe, test_dir, style_list, out_dir, img_size, label):
    """Generate all style pairs."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Collect source images
    sources = []
    for sty in style_list:
        sd = test_dir / sty
        if not sd.exists():
            continue
        for f in sorted(sd.iterdir()):
            if f.suffix.lower() in {".jpg", ".jpeg", ".png"}:
                sources.append((sty, f))

    # Pick style references (first image per target style)
    style_refs = {}
    for sty in style_list:
        sd = test_dir / sty
        if sd.exists():
            refs = sorted([f for f in sd.iterdir() if f.suffix.lower() in {".jpg", ".jpeg", ".png"}])
            if refs:
                style_refs[sty] = refs[0]

    total = len(sources) * len(style_list)
    print(f"  [{label}] {len(sources)} src x {len(style_list)} tgt = {total} pairs", flush=True)

    generator = torch.Generator(device=DEVICE).manual_seed(SEED)
    count = 0
    t0 = time.time()

    for src_style, src_path in sources:
        stem = src_path.stem
        if "__" in stem:
            stem = stem.split("__", 1)[1]
        src_img = Image.open(src_path).convert("RGB")
        if src_img.size != (img_size, img_size):
            src_img = src_img.resize((img_size, img_size), Image.LANCZOS)

        for tgt_style in style_list:
            out_name = f"{src_style}__{stem}__to__{tgt_style}.png"
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
                    prompt=PROMPT, image=src_img, ip_adapter_image=ref_img,
                    strength=STRENGTH, num_inference_steps=STEPS,
                    guidance_scale=GUIDANCE, generator=generator,
                ).images[0]
            result.save(str(out_path))
            count += 1

        if count % 50 == 0:
            print(f"  [{label}] {count}/{total}  {time.time()-t0:.0f}s", flush=True)

    elapsed = time.time() - t0
    print(f"  [{label}] Done: {count} images in {elapsed/60:.1f} min", flush=True)

    meta = {
        "method": "ip_adapter_plus_sd15_img2img", "dataset": label,
        "test_dir": str(test_dir), "out_dir": str(out_dir),
        "img_size": img_size, "style_list": style_list,
        "total_pairs": total, "total_generated": count, "total_seconds": elapsed,
    }
    with open(out_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)


def run_eval(img_dir, test_dir, style_list, idt_floor, eval_dir):
    """Evaluate using run_evaluation.py."""
    img_dir = Path(img_dir)
    eval_dir = Path(eval_dir)
    eval_dir.mkdir(parents=True, exist_ok=True)
    
    # Copy images to eval_dir/images/
    import shutil
    eval_img = eval_dir / "images"
    eval_img.mkdir(exist_ok=True)
    for f in img_dir.iterdir():
        if f.suffix.lower() in (".png", ".jpg", ".jpeg") and not f.name.startswith("_"):
            dst = eval_img / f.name
            if not dst.exists():
                shutil.copy2(str(f), str(dst))
    
    style_str = ",".join(style_list)
    cmd = [
        sys.executable, str(EVAL_SCRIPT),
        str(eval_dir),
        "--reuse_generated",
        "--save_generated_images",
        "--style_subdirs", style_str,
        "--test_dir", str(test_dir),
        "--eval_only_lpips_clip_style",
        "--clip_style_idt_baseline", str(idt_floor),
    ]
    print(f"  Running eval: {cmd[:5]} ...", flush=True)
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
    print(result.stdout[-500:] if result.stdout else "(no stdout)")
    if result.returncode != 0:
        print(f"  EVAL ERROR: {result.stderr[-500:]}")
    else:
        # Read summary
        summary_path = eval_dir / "summary.json"
        if summary_path.exists():
            with open(summary_path) as f:
                summary = json.load(f)
            cs = summary.get("clip_style", "N/A")
            lp = summary.get("content_lpips", "N/A")
            print(f"  RESULTS: CLIP-S={cs:.4f}, LPIPS={lp:.4f}" if isinstance(cs, float) else f"  RESULTS: {summary}")
    return result.returncode == 0


def main():
    print("=" * 60, flush=True)
    print("IP-Adapter: Photo2Art-256 + Random5  inference + eval", flush=True)
    print("=" * 60, flush=True)

    # ═══ Dataset 1: Photo2Art-256 ═══
    print("\n=== [1/2] Photo2Art-256 ===", flush=True)
    p2a_test = Path(r"I:\datasets\legacy256_overfit50\test")
    p2a_styles = ["Hayao", "cezanne", "monet", "photo", "vangogh"]
    p2a_out = OUT_ROOT / "photo2art256" / "images"
    
    if p2a_test.exists():
        pipe = load_pipe()
        run_inference(pipe, p2a_test, p2a_styles, p2a_out, 256, "P2A")
        del pipe; gc.collect(); torch.cuda.empty_cache(); time.sleep(3)
        # Evaluate
        ok = run_eval(p2a_out, p2a_test, p2a_styles, 0.6630, OUT_ROOT / "photo2art256" / "eval")
        print(f"  Photo2Art eval: {'OK' if ok else 'FAILED'}", flush=True)
    else:
        print("  SKIP: Photo2Art test dir not found", flush=True)

    # ═══ Dataset 2: Random5-WikiArt ═══
    print("\n=== [2/2] Random5-WikiArt ===", flush=True)
    r5_test = Path(r"I:\datasets\wikiarts20_512_test")
    # Use same 5 Distinct5 styles evaluated on wikiarts20 data pool
    # (matches SaMam Random5 protocol)
    r5_styles = ["Early_Renaissance", "Impressionism", "Minimalism", "Rococo", "Ukiyo_e"]
    r5_out = OUT_ROOT / "random5" / "images"
    
    if r5_test.exists():
        pipe = load_pipe()
        run_inference(pipe, r5_test, r5_styles, r5_out, 512, "R5")
        del pipe; gc.collect(); torch.cuda.empty_cache(); time.sleep(3)
        # Evaluate
        ok = run_eval(r5_out, r5_test, r5_styles, 0.7312, OUT_ROOT / "random5" / "eval")
        print(f"  Random5 eval: {'OK' if ok else 'FAILED'}", flush=True)
    else:
        print("  SKIP: Random5 test dir not found", flush=True)

    print("\n" + "=" * 60, flush=True)
    print("All done!", flush=True)


if __name__ == "__main__":
    main()
