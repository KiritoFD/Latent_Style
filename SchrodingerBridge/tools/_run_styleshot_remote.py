"""StyleShot batch inference - fp32 + sequential CPU offload (stable, proven approach).
All dtype fp32, sequential offload manages pipe modules.
5x5 group inference: Phase 1 (1 content/pair=20 imgs) then Phase 2 (full 30 content/pair).
"""
import os
import sys
import gc

# Monkey-patch diffusers for newer versions BEFORE any StyleShot imports
import diffusers
import diffusers.models
if not hasattr(diffusers.models, 'controlnet') or not hasattr(diffusers.models.controlnet, 'ControlNetModel'):
    from diffusers import ControlNetModel as _CNM
    import types
    _mod = types.ModuleType('diffusers.models.controlnet')
    _mod.ControlNetModel = _CNM
    sys.modules['diffusers.models.controlnet'] = _mod
    diffusers.models.controlnet = _mod

# Patch retrieve_timesteps if missing
try:
    from diffusers.pipelines.controlnet.pipeline_controlnet import retrieve_timesteps
except ImportError:
    try:
        from diffusers.pipelines.pipeline_utils import retrieve_timesteps
    except ImportError:
        from diffusers.schedulers.scheduling_utils import retrieve_timesteps
    import diffusers.pipelines.controlnet.pipeline_controlnet as _pc
    if not hasattr(_pc, 'retrieve_timesteps'):
        _pc.retrieve_timesteps = retrieve_timesteps

# Patch MultiControlNetModel if missing
try:
    from diffusers.pipelines.controlnet import MultiControlNetModel
except ImportError:
    MultiControlNetModel = type(None)

# Patch basicsr for torchvision compatibility
try:
    import torchvision.transforms.functional_tensor as _ft
except ImportError:
    import torchvision.transforms.functional as _f
    sys.modules['torchvision.transforms.functional_tensor'] = _f

import torch
import cv2
import numpy as np
from PIL import Image

# Add StyleShot to path BEFORE importing its modules
sys.path.insert(0, r"C:\Users\Administrator\StyleShot")

from annotator.hed import SOFT_HEDdetector
from diffusers import UNet2DConditionModel, ControlNetModel
from transformers import CLIPVisionModelWithProjection
from ip_adapter import StyleShot, StyleContentStableDiffusionControlNetPipeline


D5_STYLES = ["Early_Renaissance", "Impressionism", "Minimalism", "Rococo", "Ukiyo_e"]
P2A_STYLES = ["cezanne", "Hayao", "monet", "photo", "vangogh"]
R5_STYLES = ["Cubism", "Expressionism", "Pop_Art", "Romanticism", "Symbolism"]

SEED = 42
NUM_INFERENCE_STEPS = 50
GUIDANCE_SCALE = 7.5

# Remote paths
CLIP_PATH = r"I:\modelscope_cache\laion\CLIP-ViT-H-14-laion2B-s32B-b79K"
STYLESOT_WEIGHTS = r"I:\styleshot_weights\pretrained_weight"
BASE_MODEL = "runwayml/stable-diffusion-v1-5"
RESULTS_BASE = r"I:\Github\Latent_Style\SchrodingerBridge\results"

D5_CONTENT = r"I:\datasets\wikiart_distinct5_samam_512_classview\test"
D5_STYLE = r"I:\datasets\wikiart_distinct5_samam_512_classview\train"
P2A_CONTENT = r"I:\datasets\legacy256_overfit50\test"
P2A_STYLE = r"I:\datasets\legacy256_overfit50\train"
R5_BASE = r"I:\datasets\wikiarts20_512_test"
R5_CONTENT = R5_BASE
R5_STYLE = R5_BASE


def collect_content_images(content_dir, style_name=None, max_per_style=30):
    images = []
    if style_name is not None and os.path.isdir(os.path.join(content_dir, style_name)):
        for f in sorted(os.listdir(os.path.join(content_dir, style_name))):
            if f.lower().endswith(('.jpg', '.jpeg', '.png')):
                images.append(os.path.join(content_dir, style_name, f))
                if len(images) >= max_per_style:
                    break
    elif style_name is None:
        for f in sorted(os.listdir(content_dir)):
            if f.lower().endswith(('.jpg', '.jpeg', '.png')):
                images.append(os.path.join(content_dir, f))
    return images


def collect_style_images(style_dir, style_name, max_images=5):
    style_path = os.path.join(style_dir, style_name)
    images = []
    if os.path.isdir(style_path):
        for f in sorted(os.listdir(style_path)):
            if f.lower().endswith(('.jpg', '.jpeg', '.png')):
                images.append(os.path.join(style_path, f))
                if len(images) >= max_images:
                    break
    return images


def vram_mb():
    return torch.cuda.memory_allocated() / 1024**2


def run_styleshot_batch(styleshot, detector, output_dir, content_dir, style_dir,
                        styles, max_content=30, dataset_name=""):
    os.makedirs(output_dir, exist_ok=True)
    n_styles = len(styles)
    n_pairs = n_styles * n_styles  # include src==tgt (same as StyleAligned)
    total = n_pairs * max_content
    count = 0

    # Phase 1: 1 content per pair
    print(f"  [{dataset_name}] Phase 1: 1 content/pair ({n_pairs} images)...", flush=True)
    for src_style in styles:
        content_images = collect_content_images(content_dir, src_style, max_per_style=1)
        if not content_images:
            continue
        for tgt_style in styles:
            style_refs = collect_style_images(style_dir, tgt_style, max_images=1)
            if not style_refs:
                continue
            style_image = Image.open(style_refs[0]).convert("RGB")
            content_path = content_images[0]
            content_basename = os.path.splitext(os.path.basename(content_path))[0]
            out_name = f"{src_style}__{tgt_style}__{content_basename}_to_{tgt_style}.png"
            out_path = os.path.join(output_dir, out_name)

            if os.path.exists(out_path):
                count += 1
                continue

            content_bgr = cv2.imread(content_path)
            if content_bgr is None:
                continue
            content_rgb = cv2.cvtColor(content_bgr, cv2.COLOR_BGR2RGB)
            content_edge = detector(content_rgb)
            content_edge_pil = Image.fromarray(content_edge)
            # Resize to 512x512 to prevent OOM from high-res inputs
            if content_edge_pil.size != (512, 512):
                content_edge_pil = content_edge_pil.resize((512, 512), Image.LANCZOS)

            prompt = f"a painting in {tgt_style.replace('_', ' ')} style"
            try:
                result = styleshot.generate(
                    style_image=style_image,
                    prompt=[[prompt]],
                    content_image=content_edge_pil,
                    seed=SEED,
                    guidance_scale=GUIDANCE_SCALE,
                    num_inference_steps=NUM_INFERENCE_STEPS,
                )
                result[0][0].save(out_path)
                count += 1
            except Exception as e:
                print(f"  ERROR: {out_name}: {e}", flush=True)
                torch.cuda.empty_cache()
                continue

    print(f"  [{dataset_name}] Phase 1 done: {count}/{n_pairs} | VRAM={vram_mb():.0f}MB", flush=True)

    # Phase 2: Full 30 content images per pair
    print(f"  [{dataset_name}] Phase 2: Full {max_content}/pair ({total} total)...", flush=True)
    for src_style in styles:
        content_images = collect_content_images(content_dir, src_style, max_per_style=max_content)
        for tgt_style in styles:
            style_refs = collect_style_images(style_dir, tgt_style, max_images=1)
            if not style_refs:
                continue
            style_image = Image.open(style_refs[0]).convert("RGB")
            # Resize style image too
            if style_image.size != (512, 512):
                style_image = style_image.resize((512, 512), Image.LANCZOS)
            group_ok = 0
            group_skip = 0

            for content_path in content_images:
                content_basename = os.path.splitext(os.path.basename(content_path))[0]
                out_name = f"{src_style}__{tgt_style}__{content_basename}_to_{tgt_style}.png"
                out_path = os.path.join(output_dir, out_name)

                if os.path.exists(out_path):
                    group_skip += 1
                    count += 1
                    continue

                content_bgr = cv2.imread(content_path)
                if content_bgr is None:
                    continue
                content_rgb = cv2.cvtColor(content_bgr, cv2.COLOR_BGR2RGB)
                content_edge = detector(content_rgb)
                content_edge_pil = Image.fromarray(content_edge)
                # Resize to 512x512 to prevent OOM from high-res inputs
                if content_edge_pil.size != (512, 512):
                    content_edge_pil = content_edge_pil.resize((512, 512), Image.LANCZOS)

                prompt = f"a painting in {tgt_style.replace('_', ' ')} style"
                try:
                    result = styleshot.generate(
                        style_image=style_image,
                        prompt=[[prompt]],
                        content_image=content_edge_pil,
                        seed=SEED,
                        guidance_scale=GUIDANCE_SCALE,
                        num_inference_steps=NUM_INFERENCE_STEPS,
                    )
                    result[0][0].save(out_path)
                    group_ok += 1
                    count += 1
                except Exception as e:
                    print(f"  ERROR: {out_name}: {e}", flush=True)
                    torch.cuda.empty_cache()
                    continue

            print(f"  [{dataset_name}] {src_style}->{tgt_style}: +{group_ok} ok, {group_skip} skip | "
                  f"Total {count}/{total} | VRAM={vram_mb():.0f}MB", flush=True)

    print(f"  [{dataset_name}] DONE: {count}/{total}", flush=True)
    return count


def main():
    print("=" * 60)
    print("StyleShot Batch Inference (fp16 + model CPU offload)")
    print("=" * 60)

    ip_ckpt = os.path.join(STYLESOT_WEIGHTS, "ip.bin")
    style_aware_encoder_path = os.path.join(STYLESOT_WEIGHTS, "style_aware_encoder.bin")

    for p in [ip_ckpt, style_aware_encoder_path, CLIP_PATH]:
        if not os.path.exists(p):
            print(f"ERROR: Missing {p}", flush=True)
            return

    device = "cuda"
    detector = SOFT_HEDdetector()

    print("Loading StyleShot (fp16 + model CPU offload)...", flush=True)

    # Load pipe in fp16 - model_cpu_offload manages module placement
    unet = UNet2DConditionModel.from_pretrained(BASE_MODEL, subfolder="unet", torch_dtype=torch.float16)
    content_fusion_encoder = ControlNetModel.from_unet(unet).to(dtype=torch.float16)
    pipe = StyleContentStableDiffusionControlNetPipeline.from_pretrained(
        BASE_MODEL, controlnet=content_fusion_encoder, torch_dtype=torch.float16
    )

    # Delete safety_checker
    if hasattr(pipe, 'safety_checker') and pipe.safety_checker is not None:
        del pipe.safety_checker
        pipe.safety_checker = None
        gc.collect()
        print("  Deleted safety_checker", flush=True)

    # Delete feature_extractor
    if hasattr(pipe, 'feature_extractor') and pipe.feature_extractor is not None:
        del pipe.feature_extractor
        pipe.feature_extractor = None
        gc.collect()

    # Memory optimizations
    pipe.enable_attention_slicing(1)
    pipe.enable_vae_slicing()
    # model_cpu_offload: moves whole modules (UNet/VAE/text_encoder) GPU<->CPU
    # Much faster than sequential offload which moves sub-layers
    pipe.enable_model_cpu_offload()

    print(f"  Pipe configured. Init StyleShot...", flush=True)
    # StyleShot __init__ patched: no pipe.to(device), all modules fp16
    styleshot = StyleShot(device, pipe, ip_ckpt, style_aware_encoder_path, CLIP_PATH)

    print(f"Model loaded. VRAM={vram_mb():.0f}MB", flush=True)
    torch.cuda.reset_peak_memory_stats()
    print(f"  Peak VRAM during load: {torch.cuda.max_memory_allocated()/1024**2:.0f}MB", flush=True)

    # ---- D5-512 ----
    print("\n========================================")
    print(" StyleShot [D5]")
    print("========================================")
    run_styleshot_batch(
        styleshot, detector,
        output_dir=os.path.join(RESULTS_BASE, "D5-512", "styleshot"),
        content_dir=D5_CONTENT,
        style_dir=D5_STYLE,
        styles=D5_STYLES,
        max_content=30,
        dataset_name="D5-512",
    )

    # ---- P2A-256 ----
    print("\n========================================")
    print(" StyleShot [P2A]")
    print("========================================")
    run_styleshot_batch(
        styleshot, detector,
        output_dir=os.path.join(RESULTS_BASE, "P256", "styleshot"),
        content_dir=P2A_CONTENT,
        style_dir=P2A_STYLE,
        styles=P2A_STYLES,
        max_content=30,
        dataset_name="P2A-256",
    )

    # ---- R5-512 ----
    print("\n========================================")
    print(" StyleShot [R5]")
    print("========================================")
    run_styleshot_batch(
        styleshot, detector,
        output_dir=os.path.join(RESULTS_BASE, "R5-512", "styleshot"),
        content_dir=R5_CONTENT,
        style_dir=R5_STYLE,
        styles=R5_STYLES,
        max_content=30,
        dataset_name="R5-512",
    )

    peak = torch.cuda.max_memory_allocated() / 1024**2
    print(f"\nALL DONE | Peak VRAM: {peak:.0f}MB", flush=True)


if __name__ == "__main__":
    main()
