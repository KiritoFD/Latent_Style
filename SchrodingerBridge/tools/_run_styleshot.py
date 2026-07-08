"""StyleShot batch inference for D5-512, P2A-256, R5-512 datasets."""
import os
import sys

# Monkey-patch diffusers for newer versions BEFORE any StyleShot imports
import diffusers
import diffusers.models
if not hasattr(diffusers.models, 'controlnet') or not hasattr(diffusers.models.controlnet, 'ControlNetModel'):
    from diffusers import ControlNetModel as _CNM
    # Create a fake module
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

# Use HF mirror for faster downloads in China
os.environ.setdefault('HF_ENDPOINT', 'https://hf-mirror.com')

import torch
import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm
from huggingface_hub import snapshot_download

# Add StyleShot to path BEFORE importing its modules
sys.path.insert(0, r"G:\GitHub\Latent_Style\StyleShot")

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


def download_if_needed(model_id):
    """Download from HF if not already cached locally. Uses mirror."""
    from huggingface_hub import snapshot_download
    # Try offline first
    try:
        snapshot_download(model_id, local_files_only=True)
        print(f"  Using cached {model_id}", flush=True)
        return model_id
    except Exception:
        pass
    # Download with retry
    for attempt in range(3):
        try:
            print(f"  Downloading {model_id} (attempt {attempt+1})...", flush=True)
            snapshot_download(model_id, max_workers=4)
            return model_id
        except Exception as e:
            print(f"  Download failed: {e}", flush=True)
            import time; time.sleep(5)
    raise RuntimeError(f"Failed to download {model_id} after 3 attempts")


def collect_content_images(content_dir, style_name=None, max_per_style=30):
    """Collect content image paths from a directory structure.
    For D5/R5: content_dir/style_name/*.jpg
    For P2A: content_dir/*.jpg (flat)
    """
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
    """Collect style reference images from train directory."""
    style_path = os.path.join(style_dir, style_name)
    images = []
    if os.path.isdir(style_path):
        for f in sorted(os.listdir(style_path)):
            if f.lower().endswith(('.jpg', '.jpeg', '.png')):
                images.append(os.path.join(style_path, f))
                if len(images) >= max_images:
                    break
    return images


def run_styleshot_batch(styleshot, detector, output_dir, content_dir, style_dir,
                        styles, max_content=30, max_style_refs=5, dataset_name=""):
    """Run StyleShot for all style transfers."""
    os.makedirs(output_dir, exist_ok=True)
    count = 0
    total = len(styles) * (len(styles) - 1) * max_content  # exclude identity

    for src_style in styles:
        content_images = collect_content_images(content_dir, src_style, max_content)
        for tgt_style in styles:
            if src_style == tgt_style:
                continue  # skip identity transfer
            style_refs = collect_style_images(style_dir, tgt_style, max_style_refs)
            if not style_refs:
                print(f"  WARNING: no style ref for {tgt_style}, skipping", flush=True)
                continue

            # Use the first style reference image
            style_image = Image.open(style_refs[0]).convert("RGB")

            for content_path in content_images:
                # Generate output filename: Style__Style__artist__to__TargetStyle format
                content_basename = os.path.splitext(os.path.basename(content_path))[0]
                out_name = f"{src_style}__{tgt_style}__{content_basename}_to_{tgt_style}.png"
                out_path = os.path.join(output_dir, out_name)

                if os.path.exists(out_path):
                    count += 1
                    continue

                # Process content image through HED detector
                content_bgr = cv2.imread(content_path)
                if content_bgr is None:
                    continue
                content_rgb = cv2.cvtColor(content_bgr, cv2.COLOR_BGR2RGB)
                content_edge = detector(content_rgb)
                content_edge_pil = Image.fromarray(content_edge)

                # Generate
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
                except Exception as e:
                    print(f"  ERROR: {out_name}: {e}", flush=True)
                    continue

                count += 1
                if count % 25 == 0:
                    vram = torch.cuda.memory_allocated() / 1024**2
                    print(f"  [{dataset_name}] {count}/{total}  VRAM={vram:.0f}MB", flush=True)

    print(f"  [{dataset_name}] DONE: {count} images saved to {output_dir}", flush=True)
    return count


def main():
    print("=" * 60)
    print("StyleShot Batch Inference")
    print("=" * 60)

    # Use locally downloaded model weights
    base_model_path = "runwayml/stable-diffusion-v1-5"  # HF cache
    transformer_block_path = r"G:\modelscope_cache\laion\CLIP-ViT-H-14-laion2B-s32B-b79K"
    styleshot_weights_dir = r"G:\styleshot_weights\pretrained_weight"

    device = "cuda"
    detector = SOFT_HEDdetector()

    ip_ckpt = os.path.join(styleshot_weights_dir, "ip.bin")
    style_aware_encoder_path = os.path.join(styleshot_weights_dir, "style_aware_encoder.bin")

    print("Loading StyleShot model...", flush=True)
    unet = UNet2DConditionModel.from_pretrained(base_model_path, subfolder="unet")
    content_fusion_encoder = ControlNetModel.from_unet(unet)
    pipe = StyleContentStableDiffusionControlNetPipeline.from_pretrained(
        base_model_path, controlnet=content_fusion_encoder
    )
    styleshot = StyleShot(device, pipe, ip_ckpt, style_aware_encoder_path, transformer_block_path)
    print("Model loaded.", flush=True)

    # ---- D5-512 ----
    print("\n========================================")
    print(" StyleShot [D5]")
    print("========================================")
    run_styleshot_batch(
        styleshot, detector,
        output_dir=r"g:\GitHub\Latent_Style\SchrodingerBridge\results\D5-512\styleshot",
        content_dir=r"G:\GitHub\Latent_Style\Dataset\distinct5_512\test",
        style_dir=r"G:\GitHub\Latent_Style\Dataset\distinct5_512\train",
        styles=D5_STYLES,
        max_content=30,
        dataset_name="D5-512",
    )

    # ---- P2A-256 ----
    print("\n========================================")
    print(" StyleShot [P2A]")
    print("========================================")
    # P2A: content images are in test/ subfolders (cezanne, Hayao, monet, photo, vangogh)
    # style references are in train/ subfolders
    run_styleshot_batch(
        styleshot, detector,
        output_dir=r"g:\GitHub\Latent_Style\SchrodingerBridge\results\P256\styleshot",
        content_dir=r"G:\GitHub\Latent_Style\Dataset\legacy256_overfit50\test",
        style_dir=r"G:\GitHub\Latent_Style\Dataset\legacy256_overfit50\train",
        styles=P2A_STYLES,
        max_content=30,
        dataset_name="P2A-256",
    )

    # ---- R5-512 ----
    print("\n========================================")
    print(" StyleShot [R5]")
    print("========================================")
    r5_base = r"G:\GitHub\Latent_Style\Dataset\wikiart_random20_512\wikiart_random20_512\images"
    run_styleshot_batch(
        styleshot, detector,
        output_dir=r"g:\GitHub\Latent_Style\SchrodingerBridge\results\R5-512\styleshot",
        content_dir=rf"{r5_base}\test",
        style_dir=rf"{r5_base}\train",
        styles=R5_STYLES,
        max_content=30,
        dataset_name="R5-512",
    )

    print("\n" + "=" * 60)
    print("ALL DONE")
    print("=" * 60)


if __name__ == "__main__":
    main()
