"""Manual CLIP-S / LPIPS computation for IP-Adapter outputs. Avoids complex run_evaluation.py pipeline."""
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
import pyiqa

DEVICE = "cuda"

@torch.no_grad()
def compute_clip_style(gen_paths, tgt_style_pool_paths, clip_model, clip_proc):
    """CLIP-S = mean cos(CLIP(gen), pool_mean). pool = mean over target images."""
    gen_feats = []
    for p in gen_paths:
        img = Image.open(p).convert("RGB")
        feats = clip_model.get_image_features(**clip_proc(images=img, return_tensors="pt").to(DEVICE))
        gen_feats.append(feats.cpu().numpy())
    gen_feats = np.concatenate(gen_feats, axis=0)
    
    pool_feats = []
    for p in tgt_style_pool_paths:
        img = Image.open(p).convert("RGB")
        feats = clip_model.get_image_features(**clip_proc(images=img, return_tensors="pt").to(DEVICE))
        pool_feats.append(feats.cpu().numpy())
    pool_feats = np.concatenate(pool_feats, axis=0)
    pool_mean = pool_feats.mean(axis=0, keepdims=True)
    pool_mean = pool_mean / np.linalg.norm(pool_mean, axis=1, keepdims=True)
    
    gen_norm = gen_feats / np.linalg.norm(gen_feats, axis=1, keepdims=True)
    sims = (gen_norm * pool_mean).sum(axis=1)
    return float(sims.mean())

@torch.no_grad()
def compute_lpips(gen_paths, src_paths, lpips_metric):
    """Mean LPIPS between gen and source."""
    vals = []
    for gp, sp in zip(gen_paths, src_paths):
        gen = Image.open(gp).convert("RGB")
        src = Image.open(sp).convert("RGB")
        # pyiqa expects tensor
        gen_t = torch.from_numpy(np.array(gen).transpose(2,0,1)).float().unsqueeze(0).to(DEVICE) / 255.
        src_t = torch.from_numpy(np.array(src).transpose(2,0,1)).float().unsqueeze(0).to(DEVICE) / 255.
        v = lpips_metric(gen_t, src_t).item()
        vals.append(v)
    return float(np.mean(vals))

def eval_dataset(img_dir, test_dir, styles, label):
    """Evaluate CLIP-S and LPIPS for a dataset."""
    img_dir = Path(img_dir)
    test_dir = Path(test_dir)
    
    # Load CLIP
    print(f"  [{label}] Loading CLIP ViT-B/32...")
    clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(DEVICE).eval()
    clip_proc = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    
    # Load LPIPS
    print(f"  [{label}] Loading LPIPS...")
    lpips = pyiqa.create_metric("lpips", device=DEVICE)
    
    # Collect all pairs
    gen_pairs = []
    for src_style in styles:
        src_dir = test_dir / src_style
        if not src_dir.exists():
            continue
        src_files = sorted([f for f in src_dir.iterdir() if f.suffix.lower() in {".jpg", ".jpeg", ".png"}])
        for src_path in src_files:
            stem = src_path.stem
            if "__" in stem:
                stem = stem.split("__", 1)[1]
            for tgt_style in styles:
                gen_name = f"{src_style}__{stem}__to__{tgt_style}.png"
                gen_path = img_dir / gen_name
                if gen_path.exists():
                    gen_pairs.append((gen_path, src_path, tgt_style))
    
    print(f"  [{label}] Found {len(gen_pairs)} valid pairs")
    
    # CLIP-S per target style (pooled over all target style images)
    style_clip_scores = {}
    for tgt_style in styles:
        tgt_dir = test_dir / tgt_style
        tgt_pool = sorted([tgt_dir / f.name for f in tgt_dir.iterdir() if f.suffix.lower() in {".jpg", ".jpeg", ".png"}])
        pairs = [(g, s) for g, s, t in gen_pairs if t == tgt_style]
        if pairs and tgt_pool:
            g_paths = [g for g, s in pairs]
            cs = compute_clip_style(g_paths, tgt_pool, clip_model, clip_proc)
            style_clip_scores[tgt_style] = cs
            print(f"    {tgt_style}: CLIP-S={cs:.4f} ({len(g_paths)} pairs)")
    
    # Overall CLIP-S (pooled mean across all styles)
    all_tgt_pool = []
    for tgt_style in styles:
        tgt_dir = test_dir / tgt_style
        all_tgt_pool.extend([tgt_dir / f.name for f in tgt_dir.iterdir() if f.suffix.lower() in {".jpg", ".jpeg", ".png"}])
    
    all_g_paths = [g for g, s, t in gen_pairs]
    overall_cs = compute_clip_style(all_g_paths, all_tgt_pool, clip_model, clip_proc)
    
    # LPIPS (gen vs source)
    lpips_vals = []
    for g_path, src_path, _ in gen_pairs:
        try:
            v = compute_lpips([g_path], [src_path], lpips)
            lpips_vals.append(v)
        except:
            pass
    overall_lp = float(np.mean(lpips_vals))
    
    print(f"  [{label}] OVERALL: CLIP-S={overall_cs:.4f}  LPIPS={overall_lp:.4f}")
    
    del clip_model; torch.cuda.empty_cache()
    return overall_cs, overall_lp


def main():
    print("=" * 60)
    print("IP-Adapter Manual Evaluation")
    print("=" * 60)
    
    # P2A
    cs1, lp1 = eval_dataset(
        r"I:\GitHub\Latent_Style\SchrodingerBridge\exp\baseline_ipadapter\photo2art256\images",
        r"I:\datasets\legacy256_overfit50\test",
        ["Hayao", "cezanne", "monet", "photo", "vangogh"],
        "P2A-256"
    )
    
    # R5
    cs2, lp2 = eval_dataset(
        r"I:\GitHub\Latent_Style\SchrodingerBridge\exp\baseline_ipadapter\random5\images",
        r"I:\datasets\wikiarts20_512_test",
        ["Early_Renaissance", "Impressionism", "Minimalism", "Rococo", "Ukiyo_e"],
        "R5-512"
    )
    
    print("\n=== FINAL ===")
    print(f"Photo2Art-256: CLIP-S={cs1:.4f}  LPIPS={lp1:.4f}")
    print(f"Random5-WikiArt: CLIP-S={cs2:.4f}  LPIPS={lp2:.4f}")


if __name__ == "__main__":
    main()
