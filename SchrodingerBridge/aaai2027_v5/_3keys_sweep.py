"""
3 Keys Sweep v3 — Pure low-level inference (no LGTInference).
Directly loads checkpoint -> builds SpectralODEBridge620 -> calls integrate().
Bypasses ALL validation layers.
"""
import sys, os, json, time, warnings
from pathlib import Path
import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFont

warnings.filterwarnings("ignore")

ROOT = Path(os.path.abspath(__file__)).parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

# ─── Config ───────────────────────────────────────────────────────────────
CKPT_PATH   = r"I:\Github\Latent_Style\SchrodingerBridge\exp\latent256_photo2art\latent256_b16_e10\epoch_0010.pt"
CONTENT_IMG = r"I:\tmp_t2\content_input.jpg"
STYLE_ID    = 4  # vangogh
OUT_DIR     = r"I:\tmp_t2\_3keys_sweep"
DEVICE      = "cuda"

VARIANTS = {
    "Baseline": {},
    # Key 1: Zero-Step WCT tone alignment
    "K1_a0.3": {"model.zero_step_wct_enabled": True,  "model.zero_step_wct_alpha": 0.3},
    "K1_a0.5": {"model.zero_step_wct_enabled": True,  "model.zero_step_wct_alpha": 0.5},
    "K1_a0.7": {"model.zero_step_wct_enabled": True,  "model.zero_step_wct_alpha": 0.7},
    "K1_a1.0": {"model.zero_step_wct_enabled": True,  "model.zero_step_wct_alpha": 1.0},
    # Key 2: Tri-band edge preserve alpha
    "K2_epa0.3": {"bridge.tri_band_edge_preserve_alpha": 0.3},
    "K2_epa0.7": {"bridge.tri_band_edge_preserve_alpha": 0.7},
    # Key 3: AdaLN-Zero on LL
    "K3_adaln": {"model.ll_adaln_zero": True},
    # Combo K1+K3
    "Combo_K13": {"model.zero_step_wct_enabled": True, "model.zero_step_wct_alpha": 0.5,
                  "model.ll_adaln_zero": True},
}

# ══════════════════════════════════════════════════════════════════════════

def set_dot(obj, overrides):
    """Set dot-separated attributes on object."""
    for k, v in overrides.items():
        parts = k.split(".")
        target = obj
        for p in parts[:-1]:
            target = getattr(target, p)
        setattr(target, parts[-1], v)

def make_grid(imgs, cols=3, thumb=(320, 320)):
    names = list(imgs.keys())
    n = len(names); rows = (n + cols - 1) // cols
    cw, ch = thumb[0], thumb[1] + 28
    grid = Image.new("RGB", (cols*cw, rows*ch), (240,240,240))
    try:
        font = ImageFont.truetype("arial.ttf", 14)
    except:
        font = None
    for idx, name in enumerate(names):
        r, c = divmod(idx, cols); xo, yo = c*cw, r*ch
        grid.paste(imgs[name].resize(thumb, Image.LANCZOS), (xo, yo))
        d = ImageDraw.Draw(grid)
        d.rectangle([xo, yo+thumb[1], xo+cw, yo+ch], fill=(40,40,40))
        if font:
            bb = d.textbbox((0,0), name, font=font); tw = bb[2]-bb[0]
            d.text((xo+(cw-tw)//2, yo+thumb[1]+6), name, fill=(255,255,255), font=font)
        else:
            d.text((xo+4, yo+thumb[1]+6), name, fill=(255,255,255))
    return grid

def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    print("=" * 60); print("3 Keys Sweep v3 — pure low-level"); print("=" * 60)

    # ── 1. VAE ──
    print("\n[1/4] VAE...")
    from utils.inference import load_vae, encode_image, decode_latent, tensor_to_pil
    vae = load_vae(device=DEVICE, cache_dir=r"/mnt/i/Github/Latent_Style/eval_cache/hf")

    # ── 2. Content latent ──
    print("[2/4] Encoding content...")
    cimg = Image.open(CONTENT_IMG).convert("RGB").resize((256,256))
    x_np = np.array(cimg).astype(np.float32) / 255.0
    x_t = torch.from_numpy(x_np).permute(2,0,1).unsqueeze(0) * 2.0 - 1.0
    z0 = encode_image(vae, x_t, device=DEVICE)
    cimg.save(os.path.join(OUT_DIR, "_content.jpg"))
    print(f"  z0: {z0.shape}")

    # ── 3. Load model DIRECTLY (bypass LGTInference validation hell) ──
    print("[3/4] Loading model...")
    from src.spectral_bridge620 import build_spectral_ode_bridge_from_config as build_bridge
    from src.model620 import build_model_from_config
    from src.config_schema import ExperimentConfig

    ckpt = torch.load(CKPT_PATH, map_location="cpu", weights_only=False)
    raw = ckpt.get("config", {}) or {}

    # Patch objective_mode so ExperimentConfig.from_mapping doesn't explode
    if isinstance(raw.get("bridge"), dict):
        raw["bridge"]["objective_mode"] = "i2sb_endpoint"

    config = ExperimentConfig.from_mapping(raw)

    # Build model & load weights
    model = build_model_from_config(config.model, bridge_cfg=config.bridge, use_checkpointing=False).to(DEVICE)

    # Strip compile prefixes from state dict
    sd = ckpt["model_state_dict"]
    new_sd = {}
    for k, v in sd.items():
        nk = k.replace("_orig.", "").replace("._log_forward.", ".")
        new_sd[nk] = v
    # Load with non-strict to be safe
    missing, unexpected = model.load_state_dict(new_sd, strict=False)
    if missing:
        print(f"  WARNING: {len(missing)} missing keys")
    if unexpected:
        print(f"  WARNING: {len(unexpected)} unexpected keys")
    model.eval()
    print(f"  Model loaded OK.")

    # Get bridge reference
    bridge = model.bridge  # SpectralODEBridge620

    # Also need style_latent for endpoint AdaIN — get from a random target image
    # For simplicity, use zeros or sample from prior; the key is we need *some* style_latent tensor
    # Actually, integrate_transport reads style_latent from args. Let's check if it needs it.
    # Looking at code: zero_step_wct uses style_latent, endpoint_adain also.
    # We'll generate a dummy one from the style_id via DINO patches (handled inside model)
    # For now pass None and see if it works

    # ── 4. Run variants ──
    print("\n[4/4] Running variants...")
    results = {}

    for vname, voverrides in VARIANTS.items():
        print(f"  [{vname}] ", end="", flush=True)
        t0 = time.time()
        try:
            # Set overrides
            model_overrides = {k.replace("model.",""):v for k,v in voverrides.items() if k.startswith("model.")}
            bridge_overrides = {k.replace("bridge.",""):v for k,v in voverrides.items() if k.startswith("bridge.")}

            if model_overrides:
                set_dot(model, model_overrides)
            if bridge_overrides:
                set_dot(bridge, bridge_overrides)

            # Re-build bridge internals if needed (for zero_step_wct etc.)
            # The bridge reads these at integrate_transport time, not at init time
            out_z = model.integrate(z0.clone(), style_id=STYLE_ID, num_steps=8)
            img_t = decode_latent(vae, out_z, device=DEVICE)
            pil_img = tensor_to_pil(img_t[0])
            dt = time.time() - t0
            results[vname] = pil_img
            print(f"{dt:.2f}s OK")
        except Exception as e:
            print(f"ERR: {e}")
            import traceback; traceback.print_exc()

    # ── 5. Save ──
    print(f"\n--- Saving to {OUT_DIR} ---")
    for name, img in results.items():
        img.save(os.path.join(OUT_DIR, f"{name}.png"))
        print(f"  {name}.png")

    gp = os.path.join(OUT_DIR, "_comparison_grid.png")
    make_grid(results).save(gp)
    print(f"\nGRID: {gp}")
    print(f"DONE ({len(results)} variants).")

if __name__ == "__main__":
    main()
