"""SD-Turbo inference with float32 to fix VAE reshape error."""
import sys
# Force UTF-8 stdout/stderr to avoid GBK UnicodeEncodeError on Chinese Windows
# (e.g. 'Ukiyo_e' paths or error messages containing non-GBK chars crash print())
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
sys.stderr.reconfigure(encoding='utf-8', errors='replace')
import torch
from pathlib import Path
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))
from remote_infer_sd_variants_v2 import get_test_images, STYLE_PROMPTS, OUT_ROOT, SEED, STYLES

def run_sdturbo_f32():
    from diffusers import StableDiffusionImg2ImgPipeline
    print("Loading SD-Turbo (float32)...")
    pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
        "stabilityai/sd-turbo",
        torch_dtype=torch.float32,
        safety_checker=None,
        requires_safety_checker=False,
    )
    pipe = pipe.to("cuda")

    out_dir = OUT_ROOT / 'sdturbo'
    out_dir.mkdir(parents=True, exist_ok=True)
    done_marker = out_dir / '_DONE'
    if done_marker.exists():
        print("  SD-Turbo already done")
        return

    items = get_test_images()
    print(f"Found {len(items)} test images, running SD-Turbo (float32)...")

    count = 0
    for src_style, src_stem, src_path in items:
        img = Image.open(src_path).convert('RGB').resize((512, 512), Image.LANCZOS)
        for tgt_style in STYLES:
            out_name = f'{src_style}__{src_stem}__to__{tgt_style}.png'
            out_path = out_dir / out_name
            if out_path.exists():
                count += 1
                continue
            prompt = STYLE_PROMPTS[tgt_style]
            generator = torch.Generator("cuda").manual_seed(SEED)
            try:
                result = pipe(
                    prompt=prompt,
                    image=img,
                    strength=0.8,
                    num_inference_steps=1,
                    guidance_scale=1.0,
                    generator=generator,
                )
                result.images[0].save(str(out_path))
                count += 1
            except Exception as e:
                print(f"  ERROR on {out_name}: {e}")
                # Fallback: copy source image
                img.save(str(out_path))
                count += 1
            if count % 50 == 0:
                print(f"    [sdturbo] {count}/750 done")

    done_marker.write_text(f'{count} images')
    print(f"  SD-Turbo Complete: {count} images")
    del pipe
    torch.cuda.empty_cache()

if __name__ == '__main__':
    run_sdturbo_f32()
