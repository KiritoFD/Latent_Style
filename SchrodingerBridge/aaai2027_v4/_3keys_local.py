"""
Local 3-Keys Sweep for WD-VF style transfer (v2: with style reference).
Tests: zero_step_wct (K1), tri_band_soft_lock (K2), ll_adaln_zero (K3)
Checkpoint: exp/630_random20_heun_5ep/epoch_0005.pt (20 styles, flow_matching)
Content:   aaai2027_v4/teaser_content_photo_vangogh.jpg
Style Ref: exp/72_fewshot/data/.../vincent-van-gogh_road-with-cypresses-1890.jpg
Target:    Post_Impressionism (Van Gogh) = index 15

Run:  python aaai2027_v4/_3keys_local.py
"""

import sys, os, json, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]  # SchrodingerBridge/
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# ============================================================
# Paths
# ============================================================
CKPT_PATH = ROOT / "exp" / "630_random20_heun_5ep" / "epoch_0005.pt"
CONTENT_IMG = ROOT / "aaai2027_v4" / "teaser_content_photo_vangogh.jpg"
STYLE_REF_IMG = ROOT / "exp" / "72_fewshot" / "data" / "5p1_shot01" / "test" / "Post_Impressionism" / "vincent-van-gogh_road-with-cypresses-1890.jpg"
OUT_DIR = ROOT / "aaai2027_v4" / "_3keys_sweep_results"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

TARGET_STYLE_NAME = "Post_Impressionism"

# ============================================================
# Experiment matrix: (name, K1_alpha, K2_alpha, K3_enabled)
# K1=zero_step_wct_alpha  K2=edge_preserve_alpha  K3=ll_adaln_zero
# ============================================================
EXPERIMENTS = [
    ("00_baseline",           0.0, 0.50, False),
    # Key 1 only - Zero-step WCT tone mapping
    ("01_k1_wct03",          0.30, 0.50, False),
    ("02_k1_wct05",          0.50, 0.50, False),
    ("03_k1_wct07",          0.70, 0.50, False),
    ("04_k1_wct10",          1.00, 0.50, False),
    # Key 2 only - Tri-band soft locking  
    ("05_k2_lock03",         0.0, 0.30, False),
    ("06_k2_lock07",         0.0, 0.70, False),
    ("07_k2_lock09",         0.0, 0.90, False),
    # Combinations
    ("08_k1w07_k2lock07",    0.70, 0.70, False),
    ("09_k1w10_k2lock03",    1.00, 0.30, False),
]


def load_model_and_config():
    """Load checkpoint, build model."""
    print(f"[LOAD] Checkpoint: {CKPT_PATH}")
    ckpt = torch.load(CKPT_PATH, map_location="cpu", weights_only=False)
    raw_cfg = ckpt.get("config", {})
    
    from config_schema import ExperimentConfig
    from model import build_model_from_config
    
    config = ExperimentConfig.from_mapping(raw_cfg)
    model = build_model_from_config(config.model, bridge_cfg=config.bridge)
    
    state_dict = ckpt["model_state_dict"]
    new_sd = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(new_sd, strict=False)
    model = model.to(DEVICE).eval()
    
    try:
        sid = config.data.style_subdirs.index(TARGET_STYLE_NAME)
    except ValueError:
        sid = 15
    
    print(f"[OK] Model loaded | styles={len(config.data.style_subdirs)} target_id={sid} ({TARGET_STYLE_NAME})")
    return model, config, sid


def load_vae_model(device="cuda"):
    from utils.inference import download_vae_with_fallback
    vae = download_vae_with_fallback("ema", device=device)
    vae.eval()
    print(f"[OK] VAE loaded")
    return vae


def encode_image_to_latent(vae, img_path):
    """Load image and encode to fp32 latent."""
    from utils.inference import encode_image
    img = Image.open(img_path).convert("RGB").resize((512, 512), Image.LANCZOS)
    tensor = torch.from_numpy(np.array(img)).permute(2, 0, 1).float() / 255.0
    tensor = tensor.unsqueeze(0)
    latent = encode_image(vae, tensor, device=DEVICE).float()  # CRITICAL: fp32 for model
    return img, latent


def run_single_experiment(model, content_latent, style_latent, style_id, k1_alpha, k2_alpha, k3_enabled):
    """Run one inference with given key settings — patches model_cfg dataclass directly."""
    
    mcfg = model.model_cfg
    bcfg = getattr(model, 'bridge_cfg', None)
    
    # Key 1: Zero-step WCT on LL (tone pre-alignment)
    mcfg.zero_step_wct_enabled = bool(k1_alpha > 0)
    mcfg.zero_step_wct_alpha = float(k1_alpha)
    
    # Key 2: Tri-band edge preserve alpha  
    if hasattr(mcfg, 'tri_band_edge_preserve_alpha'):
        mcfg.tri_band_edge_preserve_alpha = float(k2_alpha)
    if hasattr(bcfg, 'zero_step_wct_alpha'):
        bcfg.zero_step_wct_enabled = bool(k1_alpha > 0)
        bcfg.zero_step_wct_alpha = float(k1_alpha)
    
    # ---- Run integration with style_latent for WCT ----
    b = content_latent.shape[0]
    style_tensor = torch.full((b,), style_id, dtype=torch.long, device=DEVICE)
    
    with torch.no_grad():
        result_latent = model.integrate(
            content_latent,
            style_id=style_tensor,
            num_steps=1,
            step_size=1.0,
            target_style_latent=style_latent,   # KEY: pass reference for WCT
        )
    
    return result_latent


def decode_result(vae, latent):
    from utils.inference import decode_latent
    img_tensor = decode_latent(vae, latent, device=DEVICE)
    pil_img = Image.fromarray(
        (img_tensor[0].cpu().float().permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    )
    return pil_img


def make_grid(images, labels, cols=5, cell_size=(256, 256), title=""):
    rows = (len(images) + cols - 1) // cols
    w, h = cell_size
    grid = Image.new('RGB', (cols * w + (cols+1) * 4, rows * h + (rows+1) * 24 + 28), (240, 240, 240))
    draw = ImageDraw.Draw(grid)
    try:
        font = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", 13)
    except Exception:
        font = ImageFont.load_default()
    draw.text((4, 3), title, fill=(40, 40, 40), font=font)
    for idx, (img, label) in enumerate(zip(images, labels)):
        r, c = divmod(idx, cols)
        x = c * w + (c + 1) * 4
        y = r * h + (r + 1) * 24 + 28
        grid.paste(img.resize(cell_size, Image.LANCZOS), (x, y))
        bbox = draw.textbbox((0, 0), label, font=font)
        tw = bbox[2] - bbox[0]
        draw.text((x + (w - tw) // 2, y + h + 3), label, fill=(60, 60, 60), font=font)
    return grid


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    
    print("=" * 64)
    print("WD-VF 3-Keys Local Sweep v2 (with style ref)")
    print(f"Device: {DEVICE}")
    print(f"Style Ref: {STYLE_REF_IMG.name}")
    print("=" * 64)
    
    t0 = time.time()
    model, config, style_id = load_model_and_config()
    vae = load_vae_model(DEVICE)
    
    # Encode both images
    content_pil, content_latent = encode_image_to_latent(vae, CONTENT_IMG)
    style_ref_pil, style_latent = encode_image_to_latent(vae, STYLE_REF_IMG)
    
    print(f"\n[READY] Setup took {time.time()-t0:.1f}s")
    print(f"[RUNNING] {len(EXPERIMENTS)} experiments...\n")
    
    results = [("CONTENT", content_pil)]
    
    for i, (name, k1a, k2a, k3e) in enumerate(EXPERIMENTS):
        t = time.time()
        print(f"[{i+1:2d}/{len(EXPERIMENTS)}] {name:25s} K1={k1a:.2f} K2={k2a:.2f} ...", end=" ", flush=True)
        
        try:
            out_latent = run_single_experiment(
                model, content_latent, style_latent, style_id, k1a, k2a, k3e
            )
            out_img = decode_result(vae, out_latent)
            
            save_path = OUT_DIR / f"{name}.png"
            out_img.save(save_path)
            results.append((name, out_img))
            print(f"OK {time.time()-t:.1f}s")
            
        except Exception as e:
            print(f"FAIL: {e}")
            ph = Image.new('RGB', (512, 512), (40, 40, 40))
            d = ImageDraw.Draw(ph)
            d.text((8, 8), f"ERROR:\n{str(e)[:100]}", fill='red')
            results.append((name, ph))
    
    # Also show the style reference
    results.insert(1, ("STYLE REF", style_ref_pil))
    
    labels = [r[0] for r in results]
    images = [r[1] for r in results]
    
    grid = make_grid(
        images, labels, cols=5,
        title=f"WD-VF 3-Key Sweep v2 | style={TARGET_STYLE_NAME} | ckpt=630_heun_e5"
    )
    grid_path = OUT_DIR / "_comparison_grid_v2.png"
    grid.save(grid_path, quality=95)
    print(f"\n[DONE] Grid saved: {grid_path}")
    print(f"[TOTAL] {time.time()-t0:.1f}s")
    return grid_path


if __name__ == "__main__":
    main()
