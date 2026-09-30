"""SDEdit + SD-Turbo inference on distinct5_512 - run on remote server."""
import os, sys, torch
from PIL import Image
from pathlib import Path

# ── Config ──
STYLES = ['Early_Renaissance', 'Impressionism', 'Minimalism', 'Rococo', 'Ukiyo_e']
STYLE_PROMPTS = {
    'Early_Renaissance': 'a painting in Early Renaissance style',
    'Impressionism': 'a painting in Impressionist style',
    'Minimalism': 'a painting in Minimalist abstract style',
    'Rococo': 'a painting in Rococo ornamental style',
    'Ukiyo_e': 'a painting in Ukiyo-e Japanese woodblock print style',
}
TEST_DIR = Path(r'I:\wikiart_distinct5_samam_512_classview\test')
OUT_ROOT = Path(r'I:\GitHub\Latent_Style\SchrodingerBridge\exp\baseline_v2\images')
SEED = 42

def load_image(path, size=512):
    img = Image.open(path).convert('RGB')
    img = img.resize((size, size), Image.LANCZOS)
    return img

def get_test_images():
    """Return list of (style, stem, path) for all test images."""
    items = []
    for style in STYLES:
        style_dir = TEST_DIR / style
        if not style_dir.exists():
            print(f"  WARNING: {style_dir} not found, skipping")
            continue
        for f in sorted(style_dir.iterdir()):
            if f.suffix.lower() in ('.jpg', '.png', '.jpeg'):
                items.append((style, f.stem, str(f)))
    return items

def run_sdedit():
    from diffusers import StableDiffusionImg2ImgPipeline
    print("Loading SD 1.5...")
    pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        torch_dtype=torch.float16,
        safety_checker=None,
        requires_safety_checker=False,
    )
    pipe.enable_attention_slicing()
    pipe.enable_vae_slicing()
    pipe = pipe.to("cuda")

    items = get_test_images()
    print(f"Found {len(items)} test images")

    for strength in [0.10, 0.20, 0.35, 0.40]:
        out_dir = OUT_ROOT / f'sdedit_str{strength:.2f}'
        out_dir.mkdir(parents=True, exist_ok=True)
        tag = f'sdedit_str{int(strength*100):03d}'
        done_marker = out_dir / '_DONE'
        if done_marker.exists():
            print(f"  [{tag}] already done, skipping")
            continue

        print(f"  Running SDEdit strength={strength}...")
        count = 0
        for src_style, src_stem, src_path in items:
            init_img = load_image(src_path)
            for tgt_style in STYLES:
                out_name = f'{src_style}__{src_stem}__to__{tgt_style}.png'
                out_path = out_dir / out_name
                if out_path.exists():
                    count += 1
                    continue
                prompt = STYLE_PROMPTS[tgt_style]
                generator = torch.Generator("cuda").manual_seed(SEED)
                result = pipe(
                    prompt=prompt,
                    image=init_img,
                    strength=strength,
                    num_inference_steps=20,
                    guidance_scale=7.5,
                    negative_prompt="blurry, low quality, deformed",
                    generator=generator,
                )
                result.images[0].save(str(out_path))
                count += 1
                if count % 50 == 0:
                    print(f"    [{tag}] {count}/750 done")

        done_marker.write_text(f'{count} images')
        print(f"  [{tag}] Complete: {count} images")

    del pipe
    torch.cuda.empty_cache()
    print("SDEdit done, GPU memory released")

def run_sdturbo():
    from diffusers import StableDiffusionImg2ImgPipeline
    print("Loading SD-Turbo...")
    pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
        "stabilityai/sd-turbo",
        torch_dtype=torch.float16,
        safety_checker=None,
        requires_safety_checker=False,
    )
    pipe.enable_attention_slicing()
    pipe.enable_vae_slicing()
    pipe = pipe.to("cuda")

    out_dir = OUT_ROOT / 'sdturbo'
    out_dir.mkdir(parents=True, exist_ok=True)
    done_marker = out_dir / '_DONE'
    if done_marker.exists():
        print("  SD-Turbo already done, skipping")
        del pipe
        return

    items = get_test_images()
    print(f"Found {len(items)} test images, running SD-Turbo...")

    count = 0
    for src_style, src_stem, src_path in items:
        init_img = load_image(src_path)
        for tgt_style in STYLES:
            out_name = f'{src_style}__{src_stem}__to__{tgt_style}.png'
            out_path = out_dir / out_name
            if out_path.exists():
                count += 1
                continue
            prompt = STYLE_PROMPTS[tgt_style]
            generator = torch.Generator("cuda").manual_seed(SEED)
            result = pipe(
                prompt=prompt,
                image=init_img,
                strength=0.8,
                num_inference_steps=1,
                guidance_scale=1.0,
                generator=generator,
            )
            result.images[0].save(str(out_path))
            count += 1
            if count % 50 == 0:
                print(f"    [sdturbo] {count}/750 done")

    done_marker.write_text(f'{count} images')
    print(f"  SD-Turbo Complete: {count} images")

    del pipe
    torch.cuda.empty_cache()
    print("SD-Turbo done, GPU memory released")

if __name__ == '__main__':
    print("=" * 60)
    print("SDEdit + SD-Turbo Inference on distinct5_512")
    print("=" * 60)
    run_sdedit()
    run_sdturbo()
    print("ALL DONE")
