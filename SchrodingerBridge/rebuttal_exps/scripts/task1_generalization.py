"""Task 1+: Generalization Validation across VAE families and wavelet levels.

Goal: Verify WEAVE's transport field generalizes across:
  1. Wavelet depth: 1-level Haar (production) vs 2-level Haar
  2. VAE family: SD1.5 (production) vs SDXL vs FLUX

Protocol:
  - Baseline: SD1.5 VAE + 1-level Haar (production config)
  - Exp A: SD1.5 VAE + 2-level Haar (lowpass_levels=2)
  - Exp B: SDXL VAE + 1-level Haar (4 ch, different scaling)
  - Exp C: FLUX VAE + 1-level Haar (16 ch, channel adaptation needed)

Per config: 5 source/target pairs, compute DINO-S and LPIPS.

Engineering:
  - Uses the weave_gen copy (model.py patched to allow lowpass_levels override)
  - VAE scale conversion: z_model = z_vae * (model_scale / vae_scale)
  - FLUX 16ch: take first 4 channels for model, zero-pad back for decode
"""
import json
import os
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

WEAVE_GEN_ROOT = Path(r"g:\GitHub\Latent_Style\SchrodingerBridge\weave_gen")
os.chdir(WEAVE_GEN_ROOT)
sys.path.insert(0, str(WEAVE_GEN_ROOT))

from utils.inference import LGTInference, load_vae, encode_image, decode_latent
import torchvision.transforms as T

PROD_CKPT = WEAVE_GEN_ROOT / "runs" / "submission" / "canonical_oriented_epoch4" / "epoch_0004.pt"
TEST_DIR = WEAVE_GEN_ROOT / "data" / "test"
OUTPUT_DIR = WEAVE_GEN_ROOT / "exp" / "rebuttal" / "task1_generalization"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SAMPLE_PAIRS = [
    ("Early_Renaissance", "Early_Renaissance__andrea-mantegna_adoration-of-the-magi-central-panel-from-the-altarpiece.jpg", "Impressionism"),
    ("Impressionism", "Impressionism__claude-monet_rouen-cathedral-the-portal-at-midday.jpg", "Minimalism"),
    ("Minimalism", "Minimalism__agnes-martin_happy-valley-1967.jpg", "Rococo"),
    ("Rococo", "Rococo__antoine-watteau_the-delicate-musician.jpg", "Ukiyo_e"),
    ("Ukiyo_e", "Ukiyo_e__katsushika-hokusai_cargo-ship-and-wave.jpg", "Early_Renaissance"),
]


def load_image_tensor(path, size=512):
    img = Image.open(path).convert("RGB").resize((size, size))
    t = torch.from_numpy(np.array(img)).float() / 255.0
    t = t.permute(2, 0, 1).unsqueeze(0)
    return t * 2.0 - 1.0


def set_cfg(model, **overrides):
    mcfg = getattr(model, "model_cfg", None)
    if mcfg is None:
        return
    for k, v in overrides.items():
        setattr(mcfg, k, v)


def get_style_id(model, style_name):
    style_names = getattr(model, "style_names", None) or getattr(model, "_style_names", None)
    if style_names and style_name in style_names:
        return style_names.index(style_name)
    styles_sorted = sorted([d.name for d in TEST_DIR.iterdir() if d.is_dir()])
    if style_name in styles_sorted:
        return styles_sorted.index(style_name)
    raise KeyError(f"Style '{style_name}' not found")


def compute_dino_s(src_img, gen_img, model, transform, device):
    with torch.inference_mode():
        src_t = transform(src_img).unsqueeze(0).to(device)
        gen_t = transform(gen_img).unsqueeze(0).to(device)
        feat_src = model(src_t, output_hidden_states=True).last_hidden_state[:, 0, :].float()
        feat_gen = model(gen_t, output_hidden_states=True).last_hidden_state[:, 0, :].float()
        feat_src = F.normalize(feat_src, dim=-1)
        feat_gen = F.normalize(feat_gen, dim=-1)
        return float((feat_gen * feat_src).sum().item())


def compute_lpips(src_img, gen_img, lpips_model, transform, device):
    with torch.inference_mode():
        src_t = transform(src_img).to(device)
        gen_t = transform(gen_img).to(device)
        return float(lpips_model(gen_t, src_t).item())


def load_vae_by_name(vae_name, device, cache_dir):
    """Load VAE by name. Returns (vae, scaling_factor, latent_channels)."""
    if vae_name == "sd15":
        vae = load_vae(device=str(device), model_id="ema", cache_dir=cache_dir)
        sf = float(vae.config.scaling_factor)
        lc = 4
    elif vae_name == "sdxl":
        vae = load_vae(device=str(device), model_id="stabilityai/sdxl-vae", cache_dir=cache_dir)
        sf = float(vae.config.scaling_factor)
        lc = 4
    elif vae_name == "flux":
        # FLUX VAE has 16 latent channels
        from diffusers import AutoencoderKL
        flux_vae_id = "black-forest-labs/FLUX.1-schnell"
        subfolder = "vae"
        print(f"  Loading FLUX VAE from {flux_vae_id}/{subfolder}...")
        vae = AutoencoderKL.from_pretrained(
            flux_vae_id, subfolder=subfolder,
            torch_dtype=torch.bfloat16, cache_dir=cache_dir,
        ).to(device).eval()
        sf = float(vae.config.scaling_factor)
        lc = int(getattr(vae.config, "latent_channels", 16))
    else:
        raise ValueError(f"Unknown VAE: {vae_name}")
    print(f"  VAE={vae_name}: scaling_factor={sf}, latent_channels={lc}")
    return vae, sf, lc


def encode_with_vae(vae, img_tensor, device, latent_channels):
    """Encode image to latent. If latent_channels > 4, take first 4 channels."""
    with torch.no_grad():
        z = encode_image(vae, img_tensor, device)
    if z.shape[1] > 4:
        print(f"    Channel adaptation: {z.shape[1]} -> 4 (take first 4)")
        z = z[:, :4, :, :]
    return z


def decode_with_vae(vae, z, device, latent_channels):
    """Decode latent to image. If latent_channels > 4, zero-pad back."""
    with torch.no_grad():
        if latent_channels > 4 and z.shape[1] == 4:
            # Zero-pad from 4 to latent_channels
            pad = torch.zeros(z.shape[0], latent_channels - 4, z.shape[2], z.shape[3],
                              device=z.device, dtype=z.dtype)
            z = torch.cat([z, pad], dim=1)
        img = decode_latent(vae, z, device=str(device))
    return img


def run_experiment(label, vae_name, lowpass_levels, inf, vae, vae_scale, latent_channels,
                   model_scale, dino_model, dino_transform, lpips_model, lpips_transform,
                   device, pair_data, cache_dir):
    """Run one generalization experiment configuration."""
    print(f"\n{'='*70}")
    print(f"EXP: {label} (VAE={vae_name}, levels={lowpass_levels})")
    print(f"{'='*70}")

    # Configure model
    set_cfg(inf.model,
            endpoint_adain_scale=1.0,
            endpoint_adain_scale_ll=-1.0,  # production fallback
            style_extrap_alpha=0.1,        # production
            lowpass_levels=lowpass_levels,
            lowpass_basis="haar",
            solver_type="euler")

    # Scale conversion factors
    scale_in = model_scale / max(vae_scale, 1e-8)
    scale_out = vae_scale / max(model_scale, 1e-8)
    print(f"  scale_in={scale_in:.6f}, scale_out={scale_out:.6f}")

    results = []
    crashed = False
    crash_msg = ""

    for p in pair_data:
        tgt_id_tensor = torch.tensor([p["tgt_id"]], device=device, dtype=torch.long)
        try:
            # Re-encode source with this VAE
            with torch.no_grad():
                z_src = encode_with_vae(vae, p["src_img_tensor"], device, latent_channels)
                if abs(scale_in - 1.0) > 1e-4:
                    z_src = z_src * scale_in
                z_ref = encode_with_vae(vae, p["ref_img_tensor"], device, latent_channels)
                if abs(scale_in - 1.0) > 1e-4:
                    z_ref = z_ref * scale_in

            with torch.autocast('cuda', dtype=torch.bfloat16):
                with torch.no_grad():
                    z_out = inf.model.integrate(
                        z_src,
                        style_id=tgt_id_tensor,
                        num_steps=8,
                        step_size=1.0,
                        style_strength=1.0,
                        target_style_latent=z_ref,
                    )

            if not torch.isfinite(z_out).all():
                crashed = True
                crash_msg = "non-finite latent (NaN/Inf)"
                results.append({"pair": p["name"], "crashed": True, "crash_msg": crash_msg})
                break

            with torch.no_grad():
                if abs(scale_out - 1.0) > 1e-4:
                    z_out = z_out * scale_out
                img_tensor = decode_with_vae(vae, z_out, device, latent_channels)
                img_np = img_tensor.squeeze(0).permute(1, 2, 0).cpu().float().numpy()
                img_np = (img_np * 255).clip(0, 255).astype(np.uint8)
                img_pil = Image.fromarray(img_np)
                dino_s = compute_dino_s(p["src_pil"], img_pil, dino_model, dino_transform, device)
                lpips_val = compute_lpips(p["src_pil"], img_pil, lpips_model, lpips_transform, device)
            results.append({"pair": p["name"], "dino_s": dino_s, "lpips": lpips_val})
            print(f"  {p['name']}: DINO-S={dino_s:.4f}, LPIPS={lpips_val:.4f}")
        except Exception as e:
            crashed = True
            crash_msg = f"{type(e).__name__}: {str(e)[:200]}"
            results.append({"pair": p["name"], "crashed": True, "crash_msg": crash_msg})
            traceback.print_exc()
            break

    if crashed:
        mean_dino_s = None
        mean_lpips = None
        status = "CRASHED"
        print(f"  CRASHED: {crash_msg}")
    else:
        dino_vals = [r["dino_s"] for r in results if "dino_s" in r]
        lpips_vals = [r["lpips"] for r in results if "lpips" in r]
        mean_dino_s = float(np.mean(dino_vals)) if dino_vals else None
        mean_lpips = float(np.mean(lpips_vals)) if lpips_vals else None
        status = "OK"
        print(f"  MEAN: DINO-S={mean_dino_s:.4f}, LPIPS={mean_lpips:.4f}")

    return {
        "label": label, "vae": vae_name, "lowpass_levels": lowpass_levels,
        "vae_scale": vae_scale, "latent_channels": latent_channels,
        "mean_dino_s": mean_dino_s, "mean_lpips": mean_lpips,
        "status": status, "crash_msg": crash_msg,
        "pair_results": results,
    }


def main():
    print("=" * 70)
    print("Task 1+: Generalization Validation (VAE x Wavelet levels)")
    print("=" * 70)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    cache_dir = str(WEAVE_GEN_ROOT / "eval_cache" / "hf")

    # Load production checkpoint (model only, VAE loaded separately)
    print("\nLoading production checkpoint...")
    t0 = time.time()
    inf = LGTInference(str(PROD_CKPT), device=str(device), num_steps=8)
    print(f"  Loaded in {time.time()-t0:.1f}s")
    model_scale = float(getattr(inf.model, "latent_scale_factor", 0.18215))
    print(f"  model_scale (latent_scale_factor)={model_scale}")

    # Load DINOv2
    print("\nLoading DINOv2-small...")
    from transformers import AutoModel
    dino_name = "facebook/dinov2-small"
    parts = dino_name.split("/")
    repo_dir = Path(cache_dir) / "hub" / f"models--{parts[0]}--{parts[1]}"
    snap_root = repo_dir / "snapshots"
    if snap_root.exists():
        revisions = [p for p in snap_root.iterdir() if p.is_dir()]
        if revisions:
            dino_model = AutoModel.from_pretrained(str(revisions[0])).to(device).eval()
        else:
            dino_model = AutoModel.from_pretrained(dino_name, cache_dir=cache_dir).to(device).eval()
    else:
        dino_model = AutoModel.from_pretrained(dino_name, cache_dir=cache_dir).to(device).eval()
    dino_transform = T.Compose([
        T.Resize(224, interpolation=Image.BICUBIC),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    print("  DINOv2 loaded.")

    # Load LPIPS
    print("Loading LPIPS...")
    import lpips as lpips_mod
    lpips_model = lpips_mod.LPIPS(net="vgg").to(device).eval()
    lpips_transform = T.Compose([T.ToTensor(), T.Normalize([0.5]*3, [0.5]*3)])

    # Prepare pair data (with raw image tensors for VAE encoding)
    print("\nPreparing sample pairs...")
    pair_data = []
    for src_style, src_file, tgt_style in SAMPLE_PAIRS:
        src_path = TEST_DIR / src_style / src_file
        if not src_path.exists():
            continue
        tgt_dir = TEST_DIR / tgt_style
        ref_files = sorted([p for p in tgt_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"}])
        if not ref_files:
            continue
        ref_path = ref_files[0]
        src_img = load_image_tensor(src_path).to(device)
        ref_img = load_image_tensor(ref_path).to(device)
        try:
            tgt_id = get_style_id(inf.model, tgt_style)
        except KeyError:
            tgt_id = sorted([d.name for d in TEST_DIR.iterdir() if d.is_dir()]).index(tgt_style)
        pair_data.append({
            "name": f"{src_style}->{tgt_style}",
            "tgt_id": tgt_id,
            "src_img_tensor": src_img,
            "ref_img_tensor": ref_img,
            "src_pil": Image.open(src_path).convert("RGB").resize((512, 512)),
        })
        print(f"  {src_style} -> {tgt_style}: tgt_id={tgt_id}")

    if not pair_data:
        print("ERROR: no valid pairs loaded")
        sys.exit(1)

    all_results = []

    # === Baseline: SD1.5 VAE + 1-level Haar ===
    print("\n\n### Loading SD1.5 VAE (baseline) ###")
    vae_sd15, sf_sd15, lc_sd15 = load_vae_by_name("sd15", device, cache_dir)
    r = run_experiment("baseline_sd15_1level", "sd15", 1,
                       inf, vae_sd15, sf_sd15, lc_sd15, model_scale,
                       dino_model, dino_transform, lpips_model, lpips_transform,
                       device, pair_data, cache_dir)
    all_results.append(r)
    del vae_sd15
    torch.cuda.empty_cache()

    # === Exp A: SD1.5 VAE + 2-level Haar ===
    print("\n\n### Loading SD1.5 VAE (reuse) for 2-level ###")
    vae_sd15, sf_sd15, lc_sd15 = load_vae_by_name("sd15", device, cache_dir)
    r = run_experiment("sd15_2level_haar", "sd15", 2,
                       inf, vae_sd15, sf_sd15, lc_sd15, model_scale,
                       dino_model, dino_transform, lpips_model, lpips_transform,
                       device, pair_data, cache_dir)
    all_results.append(r)
    del vae_sd15
    torch.cuda.empty_cache()

    # === Exp B: SDXL VAE + 1-level Haar ===
    print("\n\n### Loading SDXL VAE ###")
    try:
        vae_sdxl, sf_sdxl, lc_sdxl = load_vae_by_name("sdxl", device, cache_dir)
        r = run_experiment("sdxl_1level", "sdxl", 1,
                           inf, vae_sdxl, sf_sdxl, lc_sdxl, model_scale,
                           dino_model, dino_transform, lpips_model, lpips_transform,
                           device, pair_data, cache_dir)
        all_results.append(r)
        del vae_sdxl
        torch.cuda.empty_cache()
    except Exception as e:
        print(f"  SDXL VAE FAILED: {e}")
        all_results.append({"label": "sdxl_1level", "status": "VAE_LOAD_FAILED", "crash_msg": str(e)[:200]})

    # === Exp C: FLUX VAE + 1-level Haar (16ch -> 4ch adaptation) ===
    print("\n\n### Loading FLUX VAE ###")
    try:
        vae_flux, sf_flux, lc_flux = load_vae_by_name("flux", device, cache_dir)
        r = run_experiment("flux_1level_chadapt", "flux", 1,
                           inf, vae_flux, sf_flux, lc_flux, model_scale,
                           dino_model, dino_transform, lpips_model, lpips_transform,
                           device, pair_data, cache_dir)
        all_results.append(r)
        del vae_flux
        torch.cuda.empty_cache()
    except Exception as e:
        print(f"  FLUX VAE FAILED: {e}")
        all_results.append({"label": "flux_1level_chadapt", "status": "VAE_LOAD_FAILED", "crash_msg": str(e)[:200]})

    # === Summary ===
    print("\n" + "=" * 70)
    print("GENERALIZATION VALIDATION SUMMARY")
    print("=" * 70)
    print(f"  {'Label':<28} {'VAE':<8} {'Levels':<8} {'DINO-S':<10} {'LPIPS':<10} {'Status':<15}")
    for r in all_results:
        ds = f"{r['mean_dino_s']:.4f}" if r.get('mean_dino_s') is not None else "N/A"
        lp = f"{r['mean_lpips']:.4f}" if r.get('mean_lpips') is not None else "N/A"
        print(f"  {r['label']:<28} {r.get('vae','?'):<8} {r.get('lowpass_levels','?'):<8} {ds:<10} {lp:<10} {r['status']:<15}")

    # Save results
    out_path = OUTPUT_DIR / "task1_generalization_results.json"
    out_path.write_text(json.dumps(all_results, indent=2), encoding="utf-8")
    print(f"\nResults saved to: {out_path}")

    # Compute delta from baseline
    baseline = all_results[0] if all_results else None
    if baseline and baseline.get("mean_dino_s") is not None:
        print(f"\nDelta from baseline ({baseline['label']}):")
        print(f"  {'Label':<28} {'ΔDINO-S':<12} {'ΔLPIPS':<12}")
        for r in all_results[1:]:
            if r.get("mean_dino_s") is not None:
                dd = r["mean_dino_s"] - baseline["mean_dino_s"]
                dl = r["mean_lpips"] - baseline["mean_lpips"]
                print(f"  {r['label']:<28} {dd:+.4f}      {dl:+.4f}")
            else:
                print(f"  {r['label']:<28} CRASHED")

    print("TASK1_GEN_EXIT=0")


if __name__ == "__main__":
    main()
