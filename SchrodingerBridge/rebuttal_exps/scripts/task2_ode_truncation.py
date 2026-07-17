"""Task 2: ODE Truncation Error Phase-Transition Scan.

Goal: Quantitatively verify that K=8 Euler has passed the truncation error
phase-transition point of the WEAVE flow-matching ODE.

Protocol:
  1. Pure-ODE (endpoint_adain_scale=0):
     - Pseudo ground truth: RK4 K=64 endpoint z_64*
     - Sweep Euler K in {2,4,8,16,32}
     - Trajectory drift eps(K) = ||z_K - z_64*||_2^2 / ||z_64*||_2^2
     - Locate the inflection point (2nd derivative zero) of eps(K)
  2. Production pipeline (endpoint_adain_scale=2.0):
     - Sweep Euler K in {2,4,8,16,32,64}
     - Decode to image, compute DINO-S and LPIPS
     - Show metric convergence as K grows

Engineering constraints:
  - Local RTX 3060 12GB; small pair count (5 pairs) to fit budget
  - Reuse production checkpoint (epoch_0004.pt)
  - Reuse model.integrate() to avoid duplicating solver code
"""
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

WEAVE_ROOT = Path(r"g:\GitHub\Latent_Style\WEAVE")
os.chdir(WEAVE_ROOT)
sys.path.insert(0, str(WEAVE_ROOT))

from utils.inference import LGTInference, load_vae, encode_image, decode_latent
import torchvision.transforms as T

PROD_CKPT = WEAVE_ROOT / "runs" / "submission" / "canonical_oriented_epoch4" / "epoch_0004.pt"
TEST_DIR = WEAVE_ROOT / "data" / "test"
OUTPUT_DIR = WEAVE_ROOT / "exp" / "rebuttal" / "task2_ode_truncation"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# 5 source images spread across styles + 1 target style for AdaIN reference
SAMPLE_PAIRS = [
    ("Early_Renaissance", "Early_Renaissance__andrea-mantegna_adoration-of-the-magi-central-panel-from-the-altarpiece.jpg", "Impressionism"),
    ("Impressionism", "Impressionism__claude-monet_rouen-cathedral-the-portal-at-midday.jpg", "Minimalism"),
    ("Minimalism", "Minimalism__agnes-martin_happy-valley-1967.jpg", "Rococo"),
    ("Rococo", "Rococo__antoine-watteau_the-delicate-musician.jpg", "Ukiyo_e"),
    ("Ukiyo_e", "Ukiyo_e__katsushika-hokusai_cargo-ship-and-wave.jpg", "Early_Renaissance"),
]

K_SWEEP_PURE = [2, 4, 8, 16, 32]
K_SWEEP_FULL = [2, 4, 8, 16, 32, 64]
K_GT = 64  # RK4 pseudo-ground-truth


def load_image_tensor(path, size=512):
    img = Image.open(path).convert("RGB").resize((size, size))
    t = torch.from_numpy(np.array(img)).float() / 255.0
    t = t.permute(2, 0, 1).unsqueeze(0)
    return t * 2.0 - 1.0


def set_cfg(model, **overrides):
    """Override model_cfg attributes for the experiment."""
    mcfg = getattr(model, "model_cfg", None)
    if mcfg is None:
        print("WARNING: model_cfg is None; overrides may not take effect")
        return
    for k, v in overrides.items():
        setattr(mcfg, k, v)


def get_style_id(model, style_name):
    """Map style name -> style id via the model's style vocabulary."""
    # The model exposes style names through model.style_names (typical) or via tokenizer
    style_names = getattr(model, "style_names", None) or getattr(model, "_style_names", None)
    if style_names and style_name in style_names:
        return style_names.index(style_name)
    # Fallback: use the index in sorted test dir order
    styles_sorted = sorted([d.name for d in TEST_DIR.iterdir() if d.is_dir()])
    if style_name in styles_sorted:
        return styles_sorted.index(style_name)
    raise KeyError(f"Style '{style_name}' not found")


def compute_dino_s(src_img, gen_img, model, transform, device):
    """DINO-S: max cosine similarity between gen CLS and ref pool CLS."""
    with torch.inference_mode():
        src_t = transform(src_img).unsqueeze(0).to(device)
        gen_t = transform(gen_img).unsqueeze(0).to(device)
        feat_src = model(src_t, output_hidden_states=True).last_hidden_state[:, 0, :].float()
        feat_gen = model(gen_t, output_hidden_states=True).last_hidden_state[:, 0, :].float()
        feat_src = F.normalize(feat_src, dim=-1)
        feat_gen = F.normalize(feat_gen, dim=-1)
        # DINO-S = cos(gen, src) for identity preservation
        return float((feat_gen * feat_src).sum().item())


def compute_lpips(src_img, gen_img, lpips_model, transform, device):
    with torch.inference_mode():
        src_t = transform(src_img).to(device)
        gen_t = transform(gen_img).to(device)
        return float(lpips_model(gen_t, src_t).item())


def main():
    print("=" * 70)
    print("Task 2: ODE Truncation Error Phase-Transition Scan")
    print("=" * 70)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load VAE + inference model
    print("\nLoading VAE and production checkpoint...")
    t0 = time.time()
    vae = load_vae(device=str(device))
    inf = LGTInference(str(PROD_CKPT), device=str(device), num_steps=8)
    print(f"  Loaded in {time.time()-t0:.1f}s")

    model_scale = float(getattr(inf.model, "latent_scale_factor", 0.18215))
    vae_scale = float(getattr(getattr(vae, "config", None), "scaling_factor", model_scale))
    scale_in = model_scale / max(vae_scale, 1e-8)
    scale_out = vae_scale / max(model_scale, 1e-8)
    print(f"  model_scale={model_scale:.6f}, vae_scale={vae_scale:.6f}")

    # Save current cfg values so we can restore them
    mcfg = inf.model.model_cfg
    orig_solver = getattr(mcfg, "solver_type", "euler")
    orig_adain = getattr(mcfg, "endpoint_adain_scale", 0.0)
    print(f"  Original: solver_type={orig_solver}, endpoint_adain_scale={orig_adain}")

    # Load DINOv2 for DINO-S computation
    print("\nLoading DINOv2-small for DINO-S...")
    from transformers import AutoModel
    HF_CACHE = "exp/eval_cache/hf"
    dino_name = "facebook/dinov2-small"
    parts = dino_name.split("/")
    repo_dir = Path(HF_CACHE) / "hub" / f"models--{parts[0]}--{parts[1]}"
    snap_root = repo_dir / "snapshots"
    if snap_root.exists():
        revisions = [p for p in snap_root.iterdir() if p.is_dir()]
        if revisions:
            dino_model = AutoModel.from_pretrained(str(revisions[0])).to(device).eval()
        else:
            dino_model = AutoModel.from_pretrained(dino_name, cache_dir=HF_CACHE).to(device).eval()
    else:
        dino_model = AutoModel.from_pretrained(dino_name, cache_dir=HF_CACHE).to(device).eval()
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

    # Prepare source + target style latents
    print("\nEncoding source images and target style latents...")
    pair_data = []
    for src_style, src_file, tgt_style in SAMPLE_PAIRS:
        src_path = TEST_DIR / src_style / src_file
        if not src_path.exists():
            print(f"  SKIP: {src_path} missing")
            continue
        # Target style reference (first image of target style)
        tgt_dir = TEST_DIR / tgt_style
        ref_files = sorted([p for p in tgt_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"}])
        if not ref_files:
            print(f"  SKIP: no refs in {tgt_dir}")
            continue
        ref_path = ref_files[0]
        src_img = load_image_tensor(src_path).to(device)
        ref_img = load_image_tensor(ref_path).to(device)
        with torch.no_grad():
            z_src = encode_image(vae, src_img, device)
            if abs(scale_in - 1.0) > 1e-4:
                z_src = z_src * scale_in
            z_ref = encode_image(vae, ref_img, device)
            if abs(scale_in - 1.0) > 1e-4:
                z_ref = z_ref * scale_in
        try:
            tgt_id = get_style_id(inf.model, tgt_style)
        except KeyError:
            tgt_id = sorted([d.name for d in TEST_DIR.iterdir() if d.is_dir()]).index(tgt_style)
        pair_data.append({
            "src_style": src_style,
            "src_file": src_file,
            "tgt_style": tgt_style,
            "src_path": str(src_path),
            "ref_path": str(ref_path),
            "z_src": z_src,
            "z_ref": z_ref,
            "tgt_id": tgt_id,
            "src_pil": Image.open(src_path).convert("RGB").resize((512, 512)),
        })
        print(f"  {src_style} -> {tgt_style}: src={src_file}, tgt_id={tgt_id}")

    if not pair_data:
        print("ERROR: no valid pairs loaded")
        sys.exit(1)

    # ==============================
    # Part 1: Pure-ODE truncation error
    # ==============================
    print("\n" + "=" * 70)
    print("Part 1: Pure-ODE truncation error (endpoint_adain_scale=0)")
    print("=" * 70)

    set_cfg(inf.model, endpoint_adain_scale=0.0, solver_type="rk4")

    # Compute pseudo-ground-truth z_64* with RK4
    print(f"\nComputing pseudo-ground-truth: RK4 K={K_GT}...")
    t0 = time.time()
    z_gt_list = []
    for p in pair_data:
        tgt_id_tensor = torch.tensor([p["tgt_id"]], device=device, dtype=torch.long)
        with torch.autocast('cuda', dtype=torch.bfloat16):
            with torch.no_grad():
                z_gt = inf.model.integrate(
                    p["z_src"],
                    style_id=tgt_id_tensor,
                    num_steps=K_GT,
                    step_size=1.0,
                    style_strength=1.0,
                    target_style_latent=p["z_ref"],
                )
        z_gt_list.append(z_gt)
    print(f"  RK4 K={K_GT} done in {time.time()-t0:.1f}s")

    # Switch to Euler for the sweep
    set_cfg(inf.model, solver_type="euler")

    # Sweep Euler K
    pure_results = {K: [] for K in K_SWEEP_PURE}
    for K in K_SWEEP_PURE:
        print(f"\n  Euler K={K}...")
        t0 = time.time()
        for i, p in enumerate(pair_data):
            tgt_id_tensor = torch.tensor([p["tgt_id"]], device=device, dtype=torch.long)
            with torch.autocast('cuda', dtype=torch.bfloat16):
                with torch.no_grad():
                    z_K = inf.model.integrate(
                        p["z_src"],
                        style_id=tgt_id_tensor,
                        num_steps=K,
                        step_size=1.0,
                        style_strength=1.0,
                        target_style_latent=p["z_ref"],
                    )
            # Normalized trajectory drift (use float32 for accumulation)
            z_K_f = z_K.float()
            z_gt_f = z_gt_list[i].float()
            drift = (z_K_f - z_gt_f).pow(2).sum().item()
            norm_gt = z_gt_f.pow(2).sum().item()
            eps = drift / max(norm_gt, 1e-12)
            pure_results[K].append({
                "pair": f"{p['src_style']}->{p['tgt_style']}",
                "drift": drift,
                "norm_gt": norm_gt,
                "eps_normalized": eps,
            })
        mean_eps = np.mean([r["eps_normalized"] for r in pure_results[K]])
        print(f"    K={K}: mean eps={mean_eps:.6e} ({time.time()-t0:.1f}s)")

    # Save Part 1 results immediately (in case Part 2 fails)
    pure_summary_early = {}
    for K in K_SWEEP_PURE:
        eps_vals = [r["eps_normalized"] for r in pure_results[K]]
        pure_summary_early[K] = {
            "mean_eps": float(np.mean(eps_vals)),
            "std_eps": float(np.std(eps_vals)),
            "max_eps": float(np.max(eps_vals)),
            "min_eps": float(np.min(eps_vals)),
        }
    early_save = {"pure_ode_summary": pure_summary_early, "pure_ode_per_pair": pure_results}
    (OUTPUT_DIR / "task2_pure_ode_partial.json").write_text(
        json.dumps(early_save, indent=2), encoding="utf-8"
    )
    print(f"\n  Part 1 saved to: {OUTPUT_DIR / 'task2_pure_ode_partial.json'}")

    # ==============================
    # Part 2: Full pipeline convergence
    # ==============================
    print("\n" + "=" * 70)
    print("Part 2: Full pipeline convergence (endpoint_adain_scale=2.0)")
    print("=" * 70)

    set_cfg(inf.model, endpoint_adain_scale=2.0, solver_type="euler")

    full_results = {K: [] for K in K_SWEEP_FULL}
    for K in K_SWEEP_FULL:
        print(f"\n  Euler K={K} (full pipeline)...")
        t0 = time.time()
        for p in pair_data:
            tgt_id_tensor = torch.tensor([p["tgt_id"]], device=device, dtype=torch.long)
            with torch.autocast('cuda', dtype=torch.bfloat16):
                with torch.no_grad():
                    z_K = inf.model.integrate(
                        p["z_src"],
                        style_id=tgt_id_tensor,
                        num_steps=K,
                        step_size=1.0,
                        style_strength=1.0,
                        target_style_latent=p["z_ref"],
                    )
            with torch.no_grad():
                if abs(scale_out - 1.0) > 1e-4:
                    z_K = z_K * scale_out
                img_tensor = decode_latent(vae, z_K, device=str(device))
                # Convert to float32 for numpy/PIL compatibility
                img_np = img_tensor.squeeze(0).permute(1,2,0).cpu().float().numpy()
                img_np = (img_np * 255).clip(0, 255).astype(np.uint8)
                img_pil = Image.fromarray(img_np)
                dino_s = compute_dino_s(p["src_pil"], img_pil, dino_model, dino_transform, device)
                lpips_val = compute_lpips(p["src_pil"], img_pil, lpips_model, lpips_transform, device)
            full_results[K].append({
                "pair": f"{p['src_style']}->{p['tgt_style']}",
                "dino_s": dino_s,
                "lpips": lpips_val,
            })
        mean_dino = np.mean([r["dino_s"] for r in full_results[K]])
        mean_lpips = np.mean([r["lpips"] for r in full_results[K]])
        print(f"    K={K}: mean DINO-S={mean_dino:.4f}, mean LPIPS={mean_lpips:.4f} ({time.time()-t0:.1f}s)")

    # Restore cfg
    set_cfg(inf.model, solver_type=orig_solver, endpoint_adain_scale=orig_adain)

    # ==============================
    # Save results
    # ==============================
    print("\n" + "=" * 70)
    print("Saving results...")
    print("=" * 70)

    # Aggregate pure-ODE
    pure_summary = {}
    for K in K_SWEEP_PURE:
        eps_vals = [r["eps_normalized"] for r in pure_results[K]]
        pure_summary[K] = {
            "mean_eps": float(np.mean(eps_vals)),
            "std_eps": float(np.std(eps_vals)),
            "max_eps": float(np.max(eps_vals)),
            "min_eps": float(np.min(eps_vals)),
        }
    # Inflection point: 2nd derivative of log(eps) vs log(K)
    log_K = np.log(K_SWEEP_PURE)
    log_eps = np.log([pure_summary[K]["mean_eps"] for K in K_SWEEP_PURE])
    if len(K_SWEEP_PURE) >= 3:
        d2 = np.diff(log_eps, n=2)
        inflection_idx = int(np.argmin(np.abs(d2)))  # zero crossing of 2nd derivative
        inflection_K = K_SWEEP_PURE[inflection_idx + 1]
    else:
        inflection_K = None

    full_summary = {}
    for K in K_SWEEP_FULL:
        dino_vals = [r["dino_s"] for r in full_results[K]]
        lpips_vals = [r["lpips"] for r in full_results[K]]
        full_summary[K] = {
            "mean_dino_s": float(np.mean(dino_vals)),
            "std_dino_s": float(np.std(dino_vals)),
            "mean_lpips": float(np.mean(lpips_vals)),
            "std_lpips": float(np.std(lpips_vals)),
        }

    output = {
        "protocol": {
            "pure_ode": "RK4 K=64 pseudo-ground-truth, Euler K-sweep, endpoint_adain_scale=0",
            "full_pipeline": "Euler K-sweep with endpoint_adain_scale=2.0 (production)",
            "pairs": len(pair_data),
            "K_GT": K_GT,
        },
        "pure_ode_summary": pure_summary,
        "pure_ode_per_pair": pure_results,
        "pure_ode_inflection_K": inflection_K,
        "full_pipeline_summary": full_summary,
        "full_pipeline_per_pair": full_results,
    }

    out_path = OUTPUT_DIR / "task2_results.json"
    out_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(f"\nSaved: {out_path}")

    # CSV for easy plotting
    csv_path = OUTPUT_DIR / "task2_pure_ode.csv"
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write("K,mean_eps,std_eps,max_eps,min_eps\n")
        for K in K_SWEEP_PURE:
            s = pure_summary[K]
            f.write(f"{K},{s['mean_eps']:.6e},{s['std_eps']:.6e},{s['max_eps']:.6e},{s['min_eps']:.6e}\n")
    print(f"CSV: {csv_path}")

    csv_path2 = OUTPUT_DIR / "task2_full_pipeline.csv"
    with open(csv_path2, "w", encoding="utf-8") as f:
        f.write("K,mean_dino_s,std_dino_s,mean_lpips,std_lpips\n")
        for K in K_SWEEP_FULL:
            s = full_summary[K]
            f.write(f"{K},{s['mean_dino_s']:.6f},{s['std_dino_s']:.6f},{s['mean_lpips']:.6f},{s['std_lpips']:.6f}\n")
    print(f"CSV: {csv_path2}")

    # Print summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print("\nPure-ODE truncation error (normalized trajectory drift):")
    print(f"  {'K':<6} {'mean eps':<15} {'std':<15}")
    for K in K_SWEEP_PURE:
        s = pure_summary[K]
        print(f"  {K:<6} {s['mean_eps']:<15.6e} {s['std_eps']:<15.6e}")
    print(f"  Inflection point (2nd deriv zero): K={inflection_K}")

    print("\nFull pipeline convergence:")
    print(f"  {'K':<6} {'DINO-S':<12} {'LPIPS':<12}")
    for K in K_SWEEP_FULL:
        s = full_summary[K]
        print(f"  {K:<6} {s['mean_dino_s']:<12.4f} {s['mean_lpips']:<12.4f}")

    print("\nTASK2_EXIT=0")


if __name__ == "__main__":
    main()
