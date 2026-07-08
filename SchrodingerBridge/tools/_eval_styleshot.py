"""StyleShot baseline evaluation: CLIP-S, LPIPS, MUSIQ on D5-512, P2A-256, R5-512."""
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
import pyiqa

DEVICE = "cuda"


@torch.no_grad()
def _get_image_features(clip_model, clip_proc, img):
    inputs = clip_proc(images=img, return_tensors="pt").to(DEVICE)
    out = clip_model.get_image_features(**inputs)
    if hasattr(out, "image_embeds"):
        feats = out.image_embeds
    elif isinstance(out, torch.Tensor):
        feats = out
    else:
        vision_out = clip_model.vision_model(pixel_values=inputs["pixel_values"])
        feats = clip_model.visual_projection(vision_out[1])
    return feats.cpu().numpy()


@torch.no_grad()
def compute_clip_style(gen_paths, tgt_style_pool_paths, clip_model, clip_proc):
    gen_feats = []
    for p in gen_paths:
        img = Image.open(p).convert("RGB")
        gen_feats.append(_get_image_features(clip_model, clip_proc, img))
    gen_feats = np.concatenate(gen_feats, axis=0)

    pool_feats = []
    for p in tgt_style_pool_paths:
        img = Image.open(p).convert("RGB")
        pool_feats.append(_get_image_features(clip_model, clip_proc, img))
    pool_feats = np.concatenate(pool_feats, axis=0)
    pool_mean = pool_feats.mean(axis=0, keepdims=True)
    pool_mean = pool_mean / np.linalg.norm(pool_mean, axis=1, keepdims=True)

    gen_norm = gen_feats / np.linalg.norm(gen_feats, axis=1, keepdims=True)
    sims = (gen_norm * pool_mean).sum(axis=1)
    return float(sims.mean())


@torch.no_grad()
def compute_lpips(gen_paths, src_paths, lpips_metric):
    vals = []
    for gp, sp in zip(gen_paths, src_paths):
        gen = Image.open(gp).convert("RGB")
        src = Image.open(sp).convert("RGB")
        w, h = gen.size
        src = src.resize((w, h), Image.LANCZOS)
        gen_t = torch.from_numpy(np.array(gen).transpose(2, 0, 1)).float().unsqueeze(0).to(DEVICE) / 255.
        src_t = torch.from_numpy(np.array(src).transpose(2, 0, 1)).float().unsqueeze(0).to(DEVICE) / 255.
        v = lpips_metric(gen_t, src_t).item()
        vals.append(v)
    return float(np.mean(vals))


@torch.no_grad()
def compute_musiq(gen_paths, musiq_metric):
    vals = []
    for p in gen_paths:
        img = Image.open(p).convert("RGB")
        t = torch.from_numpy(np.array(img).transpose(2, 0, 1)).float().unsqueeze(0).to(DEVICE) / 255.
        v = musiq_metric(t).item()
        vals.append(v)
    return float(np.mean(vals))


def _find_source_file(src_dirs, src_stem):
    for src_dir in src_dirs:
        if not src_dir.exists():
            continue
        for ext in [".jpg", ".jpeg", ".png"]:
            candidate = src_dir / (src_stem + ext)
            if candidate.exists():
                return candidate
    return None


def collect_pairs_d5(img_dir, test_dir, train_dir, styles):
    img_dir = Path(img_dir)
    test_dir = Path(test_dir)
    train_dir = Path(train_dir)
    pairs = []
    for gen_file in sorted(img_dir.glob("*.png")):
        stem = gen_file.stem
        if "_to_" not in stem:
            continue
        left, tgt_style = stem.rsplit("_to_", 1)
        # left format: src_style__tgt_style__src_file_stem
        # src_file_stem already includes style prefix (e.g. Early_Renaissance__artist_name)
        parts = left.split("__", 2)
        if len(parts) < 3:
            continue
        src_style = parts[0]
        src_file_stem = parts[2]
        src_path = _find_source_file(
            [train_dir / src_style, test_dir / src_style],
            src_file_stem,
        )
        if src_path is None:
            continue
        pairs.append((gen_file, src_path, tgt_style))
    return pairs


def collect_pairs_p2a(img_dir, test_dir, styles):
    img_dir = Path(img_dir)
    test_dir = Path(test_dir)
    pairs = []
    for gen_file in sorted(img_dir.glob("*.png")):
        stem = gen_file.stem
        if "_to_" not in stem:
            continue
        left, tgt_style = stem.rsplit("_to_", 1)
        # left format: src_style__tgt_style__number
        parts = left.split("__")
        if len(parts) < 3:
            continue
        src_style = parts[0]
        number = parts[-1]  # last part is the file number
        src_path = _find_source_file([test_dir / src_style], number)
        if src_path is None:
            continue
        pairs.append((gen_file, src_path, tgt_style))
    return pairs


def eval_dataset(pairs, test_dir, styles, label):
    test_dir = Path(test_dir)

    print(f"  [{label}] Loading CLIP ViT-B/32...")
    clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(DEVICE).eval()
    clip_proc = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    print(f"  [{label}] Loading LPIPS...")
    lpips_metric = pyiqa.create_metric("lpips", device=DEVICE)

    musiq_metric = None
    try:
        print(f"  [{label}] Loading MUSIQ...")
        musiq_metric = pyiqa.create_metric("musiq", device=DEVICE)
    except Exception as e:
        print(f"  [{label}] MUSIQ unavailable: {e}")

    print(f"  [{label}] Found {len(pairs)} valid pairs")

    for tgt_style in styles:
        tgt_dir = test_dir / tgt_style
        if not tgt_dir.exists():
            continue
        tgt_pool = sorted([f for f in tgt_dir.iterdir() if f.suffix.lower() in {".jpg", ".jpeg", ".png"}])
        style_pairs = [(g, s) for g, s, t in pairs if t == tgt_style]
        if style_pairs and tgt_pool:
            g_paths = [g for g, s in style_pairs]
            cs = compute_clip_style(g_paths, tgt_pool, clip_model, clip_proc)
            print(f"    {tgt_style}: CLIP-S={cs:.4f} ({len(g_paths)} pairs)")

    all_tgt_pool = []
    for tgt_style in styles:
        tgt_dir = test_dir / tgt_style
        if tgt_dir.exists():
            all_tgt_pool.extend([f for f in tgt_dir.iterdir() if f.suffix.lower() in {".jpg", ".jpeg", ".png"}])
    all_g_paths = [g for g, s, t in pairs]
    overall_cs = compute_clip_style(all_g_paths, all_tgt_pool, clip_model, clip_proc)

    g_paths = [g for g, s, t in pairs]
    s_paths = [s for g, s, t in pairs]
    overall_lp = compute_lpips(g_paths, s_paths, lpips_metric)

    overall_musiq = None
    if musiq_metric is not None:
        overall_musiq = compute_musiq(g_paths, musiq_metric)

    if overall_musiq is not None:
        print(f"  [{label}] OVERALL: CLIP-S={overall_cs:.4f}  LPIPS={overall_lp:.4f}  MUSIQ={overall_musiq:.4f}")
    else:
        print(f"  [{label}] OVERALL: CLIP-S={overall_cs:.4f}  LPIPS={overall_lp:.4f}")

    del clip_model; torch.cuda.empty_cache()
    return overall_cs, overall_lp, overall_musiq


def main():
    print("=" * 60)
    print("StyleShot Baseline Evaluation")
    print("=" * 60)

    # ---- P2A-256 ----
    p2a_styles = ["cezanne", "Hayao", "monet", "photo", "vangogh"]
    p2a_pairs = collect_pairs_p2a(
        r"g:\GitHub\Latent_Style\SchrodingerBridge\results\P256\styleshot",
        r"G:\GitHub\Latent_Style\Dataset\legacy256_overfit50\test",
        p2a_styles,
    )
    print(f"\n[P2A-256] {len(p2a_pairs)} pairs collected")
    cs1, lp1, mq1 = eval_dataset(
        p2a_pairs,
        r"G:\GitHub\Latent_Style\Dataset\legacy256_overfit50\test",
        p2a_styles,
        "P2A-256",
    )

    # ---- D5-512 ----
    d5_styles = ["Early_Renaissance", "Impressionism", "Minimalism", "Rococo", "Ukiyo_e"]
    d5_pairs = collect_pairs_d5(
        r"g:\GitHub\Latent_Style\SchrodingerBridge\results\D5-512\styleshot",
        r"G:\GitHub\Latent_Style\Dataset\distinct5_512\test",
        r"G:\GitHub\Latent_Style\Dataset\distinct5_512\train",
        d5_styles,
    )
    print(f"\n[D5-512] {len(d5_pairs)} pairs collected")
    cs2, lp2, mq2 = eval_dataset(
        d5_pairs,
        r"G:\GitHub\Latent_Style\Dataset\distinct5_512\test",
        d5_styles,
        "D5-512",
    )

    # ---- R5-512 ----
    r5_styles = ["Cubism", "Expressionism", "Pop_Art", "Romanticism", "Symbolism"]
    r5_base = r"G:\GitHub\Latent_Style\Dataset\wikiart_random20_512\wikiart_random20_512\images"
    r5_pairs = collect_pairs_d5(
        r"g:\GitHub\Latent_Style\SchrodingerBridge\results\R5-512\styleshot",
        rf"{r5_base}\test",
        rf"{r5_base}\train",
        r5_styles,
    )
    print(f"\n[R5-512] {len(r5_pairs)} pairs collected")
    cs3, lp3, mq3 = eval_dataset(
        r5_pairs,
        rf"{r5_base}\test",
        r5_styles,
        "R5-512",
    )

    print("\n" + "=" * 60)
    print("=== FINAL RESULTS ===")
    if mq1 is not None:
        print(f"P2A-256 StyleShot:  CLIP-S={cs1:.4f}  LPIPS={lp1:.4f}  MUSIQ={mq1:.4f}")
    else:
        print(f"P2A-256 StyleShot:  CLIP-S={cs1:.4f}  LPIPS={lp1:.4f}")
    if mq2 is not None:
        print(f"D5-512  StyleShot:  CLIP-S={cs2:.4f}  LPIPS={lp2:.4f}  MUSIQ={mq2:.4f}")
    else:
        print(f"D5-512  StyleShot:  CLIP-S={cs2:.4f}  LPIPS={lp2:.4f}")
    if mq3 is not None:
        print(f"R5-512  StyleShot:  CLIP-S={cs3:.4f}  LPIPS={lp3:.4f}  MUSIQ={mq3:.4f}")
    else:
        print(f"R5-512  StyleShot:  CLIP-S={cs3:.4f}  LPIPS={lp3:.4f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
