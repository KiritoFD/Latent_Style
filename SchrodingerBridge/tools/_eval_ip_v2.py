"""Manual CLIP-S / LPIPS computation. Fixed CLIP output access."""
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
import pyiqa
import gc

DEVICE = "cuda"

@torch.no_grad()
def get_clip_feats(paths, model, proc, batch=16):
    feats = []
    for i in range(0, len(paths), batch):
        imgs = [Image.open(p).convert("RGB") for p in paths[i:i+batch]]
        inp = proc(images=imgs, return_tensors="pt")["pixel_values"].to(DEVICE)
        out = model.get_image_features(pixel_values=inp)
        if hasattr(out, 'image_embeds'): out = out.image_embeds
        elif hasattr(out, 'pooler_output'): out = out.pooler_output
        feats.append(out.detach().cpu().numpy())
    return np.concatenate(feats, axis=0)

def compute_clip_style(gen_paths, tgt_pool_paths, model, proc):
    gf = get_clip_feats(gen_paths, model, proc)
    pf = get_clip_feats(tgt_pool_paths, model, proc)
    pm = pf.mean(axis=0, keepdims=True)
    pm = pm / np.linalg.norm(pm, axis=1, keepdims=True)
    gf = gf / np.linalg.norm(gf, axis=1, keepdims=True)
    return float((gf * pm).sum(axis=1).mean())

def compute_lpips_batch(gen_paths, src_paths, lpips_metric, target_size=None):
    """target_size: (H,W) to resize to before LPIPS (e.g., (512,512) for R5)."""
    vals = []
    for gp, sp in zip(gen_paths, src_paths):
        g_img = Image.open(gp).convert("RGB")
        s_img = Image.open(sp).convert("RGB")
        if target_size:
            g_img = g_img.resize((target_size[1], target_size[0]), Image.LANCZOS)
            s_img = s_img.resize((target_size[1], target_size[0]), Image.LANCZOS)
        gt = torch.from_numpy(np.array(g_img).transpose(2,0,1)).float().unsqueeze(0).to(DEVICE) / 255.
        st = torch.from_numpy(np.array(s_img).transpose(2,0,1)).float().unsqueeze(0).to(DEVICE) / 255.
        vals.append(lpips_metric(gt, st).item())
    return float(np.mean(vals))

def eval_dataset(img_dir, test_dir, styles, label):
    img_dir = Path(img_dir)
    test_dir = Path(test_dir)
    
    print(f"[{label}] Loading CLIP...")
    clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(DEVICE).eval()
    clip_proc = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    print(f"[{label}] Loading LPIPS...")
    lpips = pyiqa.create_metric("lpips", device=DEVICE)
    
    gen_pairs = []
    for src_style in styles:
        sd = test_dir / src_style
        if not sd.exists(): continue
        for src_path in sorted(sd.iterdir()):
            if src_path.suffix.lower() not in {".jpg", ".jpeg", ".png"}: continue
            stem = src_path.stem
            if "__" in stem: stem = stem.split("__", 1)[1]
            for tgt_style in styles:
                gp = img_dir / f"{src_style}__{stem}__to__{tgt_style}.png"
                if gp.exists():
                    gen_pairs.append((gp, src_path, tgt_style))
    
    print(f"[{label}] {len(gen_pairs)} valid pairs")
    
    # Per-style CLIP-S
    for ts in styles:
        td = test_dir / ts
        tgt_pool = sorted([td / f.name for f in td.iterdir() if f.suffix.lower() in {".jpg", ".jpeg", ".png"}])
        pairs = [(g, s) for g, s, t in gen_pairs if t == ts]
        if pairs and tgt_pool:
            cs = compute_clip_style([g for g, s in pairs], tgt_pool, clip_model, clip_proc)
            print(f"  {ts}: CLIP-S={cs:.4f}")
    # Overall
    all_tgt = []
    for ts in styles:
        td = test_dir / ts
        all_tgt += sorted([td / f.name for f in td.iterdir() if f.suffix.lower() in {".jpg", ".jpeg", ".png"}])
    cs = compute_clip_style([g for g, s, t in gen_pairs], all_tgt, clip_model, clip_proc)
    target_sz = (512, 512) if "512" in label or "R5" in label else (256, 256)
    lp = compute_lpips_batch([g for g, s, t in gen_pairs], [s for g, s, t in gen_pairs], lpips, target_sz)
    print(f"[{label}] CLIP-S={cs:.4f} LPIPS={lp:.4f}")
    
    del clip_model; gc.collect(); torch.cuda.empty_cache()
    return cs, lp

def main():
    print("IP-Adapter Manual Eval")
    cs1, lp1 = eval_dataset(
        r"I:\GitHub\Latent_Style\SchrodingerBridge\exp\baseline_ipadapter\photo2art256\images",
        r"I:\datasets\legacy256_overfit50\test",
        ["Hayao", "cezanne", "monet", "photo", "vangogh"], "P2A-256")
    cs2, lp2 = eval_dataset(
        r"I:\GitHub\Latent_Style\SchrodingerBridge\exp\baseline_ipadapter\random5\images",
        r"I:\datasets\wikiarts20_512_test",
        ["Early_Renaissance","Impressionism","Minimalism","Rococo","Ukiyo_e"], "R5-512")
    print(f"\n=== FINAL ===")
    print(f"P2A: CLIP-S={cs1:.4f} LPIPS={lp1:.4f}")
    print(f"R5: CLIP-S={cs2:.4f} LPIPS={lp2:.4f}")

if __name__ == "__main__":
    main()
