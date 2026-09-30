"""Task 1+ Extension: TAESD VAE (Tiny Autoencoder) generalization test.

TAESD is a tiny VAE (~1M params) with a completely different architecture
from SD1.5 VAE (~80M params), but same 4-channel latent space.
Tests whether WEAVE's transport field works with a drastically different VAE.
"""
import json, os, sys, time, traceback
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


def main():
    print("=" * 70)
    print("Task 1+ Ext: TAESD VAE generalization test")
    print("=" * 70)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cache_dir = str(WEAVE_GEN_ROOT / "eval_cache" / "hf")

    print("\nLoading production checkpoint...")
    inf = LGTInference(str(PROD_CKPT), device=str(device), num_steps=8)
    model_scale = float(getattr(inf.model, "latent_scale_factor", 0.18215))

    # Load DINOv2
    print("Loading DINOv2-small...")
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

    print("Loading LPIPS...")
    import lpips as lpips_mod
    lpips_model = lpips_mod.LPIPS(net="vgg").to(device).eval()
    lpips_transform = T.Compose([T.ToTensor(), T.Normalize([0.5]*3, [0.5]*3)])

    # Prepare pairs
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

    # Try loading TAESD
    print("\nLoading TAESD VAE...")
    try:
        from diffusers import AutoencoderTiny
        taesd = AutoencoderTiny.from_pretrained("madebyollin/taesd", cache_dir=cache_dir,
                                                 torch_dtype=torch.float32).to(device).eval()
        taesd_scale = 1.0  # TAESD has no scaling_factor config, use 1.0
        print(f"  TAESD loaded. scaling_factor={taesd_scale}, latent_channels=4")
    except Exception as e:
        print(f"  TAESD load failed: {e}")
        print("TAESD_EXT_EXIT=0")
        return

    # Configure model for production baseline
    set_cfg(inf.model,
            endpoint_adain_scale=1.0,
            endpoint_adain_scale_ll=-1.0,
            style_extrap_alpha=0.1,
            lowpass_levels=1,
            lowpass_basis="haar",
            solver_type="euler")

    # Scale conversion
    vae_scale = taesd_scale
    scale_in = model_scale / max(vae_scale, 1e-8)
    scale_out = vae_scale / max(model_scale, 1e-8)
    print(f"  scale_in={scale_in:.6f}, scale_out={scale_out:.6f}")

    results = []
    for p in pair_data:
        tgt_id_tensor = torch.tensor([p["tgt_id"]], device=device, dtype=torch.long)
        try:
            with torch.no_grad():
                # TAESD uses different encode/decode interface
                # TAESD encode expects [0,1] range, outputs latent
                z_src = taesd.encode(p["src_img_tensor"] * 0.5 + 0.5).latents  # [-1,1] -> [0,1]
                z_ref = taesd.encode(p["ref_img_tensor"] * 0.5 + 0.5).latents
                if abs(scale_in - 1.0) > 1e-4:
                    z_src = z_src * scale_in
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
                results.append({"pair": p["name"], "crashed": True, "crash_msg": "non-finite latent"})
                print(f"  {p['name']}: CRASHED (NaN/Inf)")
                continue

            with torch.no_grad():
                if abs(scale_out - 1.0) > 1e-4:
                    z_out = z_out * scale_out
                # TAESD decode expects latent, outputs [0,1] range
                dec_out = taesd.decode(z_out)
                img_tensor = dec_out.sample if hasattr(dec_out, 'sample') else dec_out
                img_np = img_tensor.squeeze(0).permute(1, 2, 0).cpu().float().numpy()
                img_np = (img_np * 255).clip(0, 255).astype(np.uint8)
                img_pil = Image.fromarray(img_np)
                dino_s = compute_dino_s(p["src_pil"], img_pil, dino_model, dino_transform, device)
                lpips_val = compute_lpips(p["src_pil"], img_pil, lpips_model, lpips_transform, device)
            results.append({"pair": p["name"], "dino_s": dino_s, "lpips": lpips_val})
            print(f"  {p['name']}: DINO-S={dino_s:.4f}, LPIPS={lpips_val:.4f}")
        except Exception as e:
            results.append({"pair": p["name"], "crashed": True, "crash_msg": str(e)[:200]})
            traceback.print_exc()

    dino_vals = [r["dino_s"] for r in results if "dino_s" in r]
    lpips_vals = [r["lpips"] for r in results if "lpips" in r]
    mean_dino = float(np.mean(dino_vals)) if dino_vals else None
    mean_lpips = float(np.mean(lpips_vals)) if lpips_vals else None
    status = "OK" if mean_dino is not None else "CRASHED"
    print(f"\n  MEAN: DINO-S={mean_dino}, LPIPS={mean_lpips}, status={status}")

    # Load existing results and append
    existing_path = OUTPUT_DIR / "task1_generalization_results.json"
    if existing_path.exists():
        all_results = json.loads(existing_path.read_text(encoding="utf-8"))
    else:
        all_results = []

    all_results.append({
        "label": "taesd_1level",
        "vae": "taesd",
        "lowpass_levels": 1,
        "vae_scale": vae_scale,
        "latent_channels": 4,
        "mean_dino_s": mean_dino,
        "mean_lpips": mean_lpips,
        "status": status,
        "pair_results": results,
    })

    existing_path.write_text(json.dumps(all_results, indent=2), encoding="utf-8")
    print(f"\nResults appended to: {existing_path}")
    print("TAESD_EXT_EXIT=0")


if __name__ == "__main__":
    main()
