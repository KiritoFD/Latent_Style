"""Task 4b: Failure Boundary in per_subband_wct mode (true 2D boundary).

Supplementary to Task 4. In the production spatial_fiber mode, lambda_LL
(endpoint_adain_scale_ll) is architecturally inert because LL is never touched
by AdaIN (fiber = h - lp(h) excludes LL). This makes the failure boundary
degenerate (1D, eta-only).

To map the TRUE 2D failure boundary where lambda_LL actually affects LL AdaIN,
we switch to per_subband_wct mode, where adain_scale_ll directly controls
WCT-based style injection into the LL subband.

Same 30-point grid as Task 4:
  lambda_LL in {0.0, 0.1, 0.3, 1.0, 3.0, 10.0}
  eta       in {0.0, 0.1, 0.5, 1.0, 2.0}

Output:
  task4b_grid_results.json   - full per-pair results
  task4b_grid_summary.csv    - per-grid-point summary
  task4b_failure_heatmap.png - DINO-C heatmap with 2D failure boundary
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

WEAVE_ROOT = Path(r"g:\GitHub\Latent_Style\WEAVE")
os.chdir(WEAVE_ROOT)
sys.path.insert(0, str(WEAVE_ROOT))

from utils.inference import LGTInference, load_vae, encode_image, decode_latent
import torchvision.transforms as T

PROD_CKPT = WEAVE_ROOT / "runs" / "submission" / "canonical_oriented_epoch4" / "epoch_0004.pt"
TEST_DIR = WEAVE_ROOT / "data" / "test"
OUTPUT_DIR = WEAVE_ROOT / "exp" / "rebuttal" / "task4_failure_boundary"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SAMPLE_PAIRS = [
    ("Early_Renaissance", "Early_Renaissance__andrea-mantegna_adoration-of-the-magi-central-panel-from-the-altarpiece.jpg", "Impressionism"),
    ("Impressionism", "Impressionism__claude-monet_rouen-cathedral-the-portal-at-midday.jpg", "Minimalism"),
    ("Minimalism", "Minimalism__agnes-martin_happy-valley-1967.jpg", "Rococo"),
    ("Rococo", "Rococo__antoine-watteau_the-delicate-musician.jpg", "Ukiyo_e"),
    ("Ukiyo_e", "Ukiyo_e__katsushika-hokusai_cargo-ship-and-wave.jpg", "Early_Renaissance"),
]

LAMBDA_LL_GRID = [0.0, 0.1, 0.3, 1.0, 3.0, 10.0]
ETA_GRID = [0.0, 0.1, 0.5, 1.0, 2.0]
DINO_C_FAILURE_THRESHOLD = 0.215


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


def compute_dino_c(src_img, gen_img, model, transform, device):
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


def main():
    print("=" * 70)
    print("Task 4b: Failure Boundary in per_subband_wct mode (true 2D)")
    print("=" * 70)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    print("\nLoading VAE and production checkpoint...")
    t0 = time.time()
    vae = load_vae(device=str(device))
    inf = LGTInference(str(PROD_CKPT), device=str(device), num_steps=8)
    print(f"  Loaded in {time.time()-t0:.1f}s")

    model_scale = float(getattr(inf.model, "latent_scale_factor", 0.18215))
    vae_scale = float(getattr(getattr(vae, "config", None), "scaling_factor", model_scale))
    scale_in = model_scale / max(vae_scale, 1e-8)
    scale_out = vae_scale / max(model_scale, 1e-8)

    print("\nLoading DINOv2-small for DINO-C...")
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

    print("Loading LPIPS...")
    import lpips as lpips_mod
    lpips_model = lpips_mod.LPIPS(net="vgg").to(device).eval()
    lpips_transform = T.Compose([T.ToTensor(), T.Normalize([0.5]*3, [0.5]*3)])

    print("\nEncoding source images and target style latents...")
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
            "src_style": src_style, "tgt_style": tgt_style,
            "z_src": z_src, "z_ref": z_ref, "tgt_id": tgt_id,
            "src_pil": Image.open(src_path).convert("RGB").resize((512, 512)),
        })
        print(f"  {src_style} -> {tgt_style}: tgt_id={tgt_id}")

    if not pair_data:
        print("ERROR: no valid pairs loaded")
        sys.exit(1)

    # Grid sweep in per_subband_wct mode
    grid_results = {}
    summary_rows = []
    total_points = len(LAMBDA_LL_GRID) * len(ETA_GRID)
    point_idx = 0

    for lam_ll in LAMBDA_LL_GRID:
        for eta in ETA_GRID:
            point_idx += 1
            key = f"ll_{lam_ll}_eta_{eta}"
            print(f"\n[{point_idx}/{total_points}] lambda_LL={lam_ll}, eta={eta} (per_subband_wct)")

            # Switch to per_subband_wct mode where adain_scale_ll is active
            set_cfg(inf.model,
                    endpoint_adain_scale=1.0,
                    endpoint_adain_scale_ll=lam_ll,
                    style_extrap_alpha=eta,
                    endpoint_adain_mode="per_subband_wct",
                    solver_type="euler")

            t0 = time.time()
            pair_results = []
            crashed = False
            crash_msg = ""
            for p in pair_data:
                tgt_id_tensor = torch.tensor([p["tgt_id"]], device=device, dtype=torch.long)
                try:
                    with torch.autocast('cuda', dtype=torch.bfloat16):
                        with torch.no_grad():
                            z_out = inf.model.integrate(
                                p["z_src"],
                                style_id=tgt_id_tensor,
                                num_steps=8,
                                step_size=1.0,
                                style_strength=1.0,
                                target_style_latent=p["z_ref"],
                            )
                    if not torch.isfinite(z_out).all():
                        crashed = True
                        crash_msg = "non-finite latent (NaN/Inf)"
                        pair_results.append({"pair": f"{p['src_style']}->{p['tgt_style']}",
                                             "crashed": True, "crash_msg": crash_msg})
                        break
                    with torch.no_grad():
                        if abs(scale_out - 1.0) > 1e-4:
                            z_out = z_out * scale_out
                        img_tensor = decode_latent(vae, z_out, device=str(device))
                        img_np = img_tensor.squeeze(0).permute(1, 2, 0).cpu().float().numpy()
                        img_np = (img_np * 255).clip(0, 255).astype(np.uint8)
                        img_pil = Image.fromarray(img_np)
                        img_arr = np.array(img_pil).astype(np.float32)
                        if img_arr.std() < 1.0:
                            crashed = True
                            crash_msg = f"degenerate image (std={img_arr.std():.3f})"
                            pair_results.append({"pair": f"{p['src_style']}->{p['tgt_style']}",
                                                 "crashed": True, "crash_msg": crash_msg})
                            break
                        dino_c = compute_dino_c(p["src_pil"], img_pil, dino_model, dino_transform, device)
                        lpips_val = compute_lpips(p["src_pil"], img_pil, lpips_model, lpips_transform, device)
                    pair_results.append({"pair": f"{p['src_style']}->{p['tgt_style']}",
                                         "dino_c": dino_c, "lpips": lpips_val})
                except Exception as e:
                    crashed = True
                    crash_msg = f"{type(e).__name__}: {str(e)[:200]}"
                    pair_results.append({"pair": f"{p['src_style']}->{p['tgt_style']}",
                                         "crashed": True, "crash_msg": crash_msg})
                    traceback.print_exc()
                    break

            elapsed = time.time() - t0
            if crashed:
                mean_dino_c = None
                mean_lpips = None
                status = "CRASHED"
                print(f"  CRASHED: {crash_msg} ({elapsed:.1f}s)")
            else:
                dino_c_vals = [r["dino_c"] for r in pair_results if "dino_c" in r]
                lpips_vals = [r["lpips"] for r in pair_results if "lpips" in r]
                mean_dino_c = float(np.mean(dino_c_vals)) if dino_c_vals else None
                mean_lpips = float(np.mean(lpips_vals)) if lpips_vals else None
                if mean_dino_c is not None and mean_dino_c < DINO_C_FAILURE_THRESHOLD:
                    status = "FAILURE_DINO_C"
                else:
                    status = "OK"
                print(f"  {status}: DINO-C={mean_dino_c}, LPIPS={mean_lpips} ({elapsed:.1f}s)")

            grid_results[key] = {
                "lambda_ll": lam_ll, "eta": eta,
                "mean_dino_c": mean_dino_c, "mean_lpips": mean_lpips,
                "status": status, "crash_msg": crash_msg,
                "pair_results": pair_results, "elapsed_s": elapsed,
            }
            summary_rows.append({
                "lambda_ll": lam_ll, "eta": eta,
                "mean_dino_c": mean_dino_c, "mean_lpips": mean_lpips,
                "status": status, "crash_msg": crash_msg,
            })

            (OUTPUT_DIR / "task4b_grid_results.json").write_text(
                json.dumps(grid_results, indent=2), encoding="utf-8")

    # Write CSV summary
    import csv
    csv_path = OUTPUT_DIR / "task4b_grid_summary.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "lambda_ll", "eta", "mean_dino_c", "mean_lpips", "status", "crash_msg"])
        writer.writeheader()
        writer.writerows(summary_rows)
    print(f"\nCSV saved: {csv_path}")

    # Print summary table
    print("\n" + "=" * 70)
    print("FAILURE BOUNDARY SUMMARY (per_subband_wct mode)")
    print("=" * 70)
    print(f"  {'lambda_LL':<10} {'eta':<6} {'DINO-C':<10} {'LPIPS':<10} {'Status':<18}")
    for row in summary_rows:
        dc = f"{row['mean_dino_c']:.4f}" if row['mean_dino_c'] is not None else "N/A"
        lp = f"{row['mean_lpips']:.4f}" if row['mean_lpips'] is not None else "N/A"
        print(f"  {row['lambda_ll']:<10} {row['eta']:<6} {dc:<10} {lp:<10} {row['status']:<18}")

    # Generate heatmap
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        dino_c_matrix = np.full((len(LAMBDA_LL_GRID), len(ETA_GRID)), np.nan)
        for i, lam in enumerate(LAMBDA_LL_GRID):
            for j, eta in enumerate(ETA_GRID):
                key = f"ll_{lam}_eta_{eta}"
                if key in grid_results and grid_results[key]["mean_dino_c"] is not None:
                    dino_c_matrix[i, j] = grid_results[key]["mean_dino_c"]

        fig, ax = plt.subplots(figsize=(8, 6))
        cmap = plt.cm.RdYlGn.copy()
        cmap.set_bad(color='#666666')
        im = ax.imshow(dino_c_matrix, cmap=cmap, vmin=0.0, vmax=0.6, aspect='auto', origin='lower')

        # Contour only if we have enough valid data
        if not np.all(np.isnan(dino_c_matrix)):
            try:
                ax.contour(np.arange(len(ETA_GRID)), np.arange(len(LAMBDA_LL_GRID)),
                           dino_c_matrix, levels=[DINO_C_FAILURE_THRESHOLD],
                           colors='red', linewidths=3, linestyles='--')
            except Exception:
                pass

        for i in range(len(LAMBDA_LL_GRID)):
            for j in range(len(ETA_GRID)):
                val = dino_c_matrix[i, j]
                if np.isnan(val):
                    txt = "CRASH"
                    color = 'white'
                else:
                    txt = f"{val:.3f}"
                    color = 'black' if val > 0.4 else 'white'
                ax.text(j, i, txt, ha='center', va='center', color=color, fontsize=9, fontweight='bold')

        ax.set_xticks(range(len(ETA_GRID)))
        ax.set_xticklabels([f"{e}" for e in ETA_GRID])
        ax.set_yticks(range(len(LAMBDA_LL_GRID)))
        ax.set_yticklabels([f"{l}" for l in LAMBDA_LL_GRID])
        ax.set_xlabel("eta (style_extrap_alpha)", fontsize=12)
        ax.set_ylabel("lambda_LL (endpoint_adain_scale_ll)", fontsize=12)
        ax.set_title("Failure Boundary (per_subband_wct): DINO-C heatmap\n(red dashed = TGT floor 0.215)", fontsize=11)
        plt.colorbar(im, ax=ax, label="DINO-C (content preservation)")
        plt.tight_layout()
        heatmap_path = OUTPUT_DIR / "task4b_failure_heatmap.png"
        plt.savefig(heatmap_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Heatmap saved: {heatmap_path}")
    except Exception as e:
        print(f"WARNING: heatmap generation failed: {e}")

    print(f"\nResults saved to: {OUTPUT_DIR}")
    print("TASK4B_EXIT=0")


if __name__ == "__main__":
    main()
