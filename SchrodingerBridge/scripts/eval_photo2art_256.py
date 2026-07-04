"""Evaluate 256 baselines on photo2art 5 styles (cezanne/Hayao/monet/photo/vangogh).

Computes CLIP-S, CLIP-T, LPIPS, MUSIQ, ART-FID for each baseline's gen_dir.

Usage:
    python eval_photo2art_256.py --methods-json methods.json --output results.json
"""
import argparse
import json
import re
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

# photo2art 5 styles
STYLE_NAMES = ["cezanne", "Hayao", "monet", "photo", "vangogh"]
STYLE_PROMPTS = {
    "cezanne": "a painting in Paul Cezanne style",
    "Hayao": "a painting in Hayao Miyazaki style",
    "monet": "a painting in Claude Monet style",
    "photo": "a photograph",
    "vangogh": "a painting in Vincent van Gogh style",
}

TEST_ROOT = Path("/mnt/i/legacy256_overfit50/test")


def parse_filename(name: str):
    """Parse {src_style}_{id}_to_{tgt_style}.jpg -> (src_style, id, tgt_style).

    Also handles __to__ format.
    """
    stem = name.rsplit(".", 1)[0] if "." in name else name

    if "__to__" in stem:
        left, tgt_style = stem.rsplit("__to__", 1)
        parts = left.split("__", 2)
        if len(parts) >= 2:
            return parts[0], parts[-1], tgt_style
        return None, None, None

    # {src_style}_{id}_to_{tgt_style}
    m = re.match(r"^(cezanne|Hayao|monet|photo|vangogh)_(.+?)_to_(cezanne|Hayao|monet|photo|vangogh)$", stem)
    if m:
        return m.group(1), m.group(2), m.group(3)

    return None, None, None


def collect_image_files(root: Path, max_images=0):
    files = sorted(list(root.glob("*.png")) + list(root.glob("*.jpg")))
    if max_images > 0 and len(files) > max_images:
        files = files[:max_images]
    return files


def load_image(path, size=256):
    img = Image.open(path).convert("RGB").resize((size, size), Image.LANCZOS)
    return img


# ---------- CLIP-S, CLIP-T ----------
def compute_clip_metrics(gen_files, device, dtype, batch_size=16):
    """CLIP-S: cos(CLIP(gen), CLIP(ref_prototype)).
    CLIP-T: cos(CLIP(gen), CLIP(text(style_prompt))).
    """
    from transformers import CLIPModel, CLIPProcessor

    print("[CLIP] Loading model...")
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device).eval()
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    # Build style reference prototypes
    ref_features = {}
    for style in STYLE_NAMES:
        style_dir = TEST_ROOT / style
        ref_files = sorted(list(style_dir.glob("*.jpg")) + list(style_dir.glob("*.png")))[:30]
        if not ref_files:
            print(f"[WARN] No ref images for {style}")
            continue
        feats = []
        for rf in ref_files:
            img = load_image(rf)
            inputs = processor(images=img, return_tensors="pt").to(device)
            with torch.no_grad():
                f = model.get_image_features(**inputs).float()
                f = F.normalize(f, dim=-1)
            feats.append(f)
        ref_features[style] = torch.cat(feats).mean(0, keepdim=True)
        ref_features[style] = F.normalize(ref_features[style], dim=-1)

    # Text prompts
    text_inputs = processor(text=[STYLE_PROMPTS[s] for s in STYLE_NAMES], return_tensors="pt", padding=True).to(device)
    with torch.no_grad():
        text_feats = model.get_text_features(**text_inputs).float()
        text_feats = F.normalize(text_feats, dim=-1)
    text_feat_map = {s: text_feats[i] for i, s in enumerate(STYLE_NAMES)}

    # Process gen files
    clip_s_list = []
    clip_t_list = []
    valid_files = []

    for i, gf in enumerate(gen_files):
        parsed = parse_filename(gf.name)
        if parsed[2] is None:
            continue
        _, _, tgt_style = parsed
        if tgt_style not in ref_features:
            continue

        img = load_image(gf)
        inputs = processor(images=img, return_tensors="pt").to(device)
        with torch.no_grad():
            gen_feat = model.get_image_features(**inputs).float()
            gen_feat = F.normalize(gen_feat, dim=-1)

        clip_s = (gen_feat * ref_features[tgt_style]).sum().item()
        clip_t = (gen_feat * text_feat_map[tgt_style]).sum().item()
        clip_s_list.append(clip_s)
        clip_t_list.append(clip_t)
        valid_files.append(gf)

    del model
    torch.cuda.empty_cache()

    return {
        "clip_s": float(np.mean(clip_s_list)) if clip_s_list else None,
        "clip_t": float(np.mean(clip_t_list)) if clip_t_list else None,
        "n_valid": len(clip_s_list),
    }


# ---------- LPIPS (content_distance) ----------
def compute_lpips(gen_files, device, batch_size=16):
    """LPIPS content distance: gen vs src (same src_style image from test set)."""
    import lpips
    print("[LPIPS] Loading model...")
    lpips_fn = lpips.LPIPS(net="alex").to(device).eval()

    lpips_list = []
    for gf in gen_files:
        parsed = parse_filename(gf.name)
        if parsed[0] is None:
            continue
        src_style, src_id, _ = parsed
        # Find corresponding src image in test set
        src_dir = TEST_ROOT / src_style
        src_file = None
        for sf in src_dir.iterdir():
            if sf.stem == src_id or sf.name == f"{src_id}.jpg":
                src_file = sf
                break
        if src_file is None:
            continue

        gen_img = load_image(gf)
        src_img = load_image(src_file)

        gen_t = torch.from_numpy(np.array(gen_img)).permute(2, 0, 1).float() / 127.5 - 1.0
        src_t = torch.from_numpy(np.array(src_img)).permute(2, 0, 1).float() / 127.5 - 1.0
        gen_t = gen_t.unsqueeze(0).to(device)
        src_t = src_t.unsqueeze(0).to(device)

        with torch.no_grad():
            d = lpips_fn(gen_t, src_t).item()
        lpips_list.append(d)

    del lpips_fn
    torch.cuda.empty_cache()

    return {"lpips": float(np.mean(lpips_list)) if lpips_list else None, "n_valid": len(lpips_list)}


# ---------- MUSIQ ----------
def compute_musiq(gen_files, device, batch_size=16):
    print("[MUSIQ] Loading model...")
    import pyiqa
    musiq = pyiqa.create_metric("musiq_koniq").to(device).eval()

    scores = []
    for gf in gen_files:
        img = load_image(gf)
        with torch.no_grad():
            s = musiq(img).item()
        scores.append(s)

    del musiq
    torch.cuda.empty_cache()

    return {"musiq": float(np.mean(scores)) if scores else None, "n_valid": len(scores)}


# ---------- ART-FID ----------
def compute_art_fid(gen_files, device, max_gen=200, max_ref=200):
    print("[ART-FID] Computing...")
    from artfid_metric import ArtFID

    art_fid = ArtFID(device=device, art_fid_weights_path=Path("src/utils/art_inception.pth"))

    # Collect gen features
    gen_imgs = []
    for gf in gen_files[:max_gen]:
        img = load_image(gf)
        gen_imgs.append(np.array(img))

    # Collect ref features (all test images)
    ref_imgs = []
    for style in STYLE_NAMES:
        style_dir = TEST_ROOT / style
        for rf in sorted(style_dir.iterdir())[:max_ref // 5]:
            img = load_image(rf)
            ref_imgs.append(np.array(img))

    result = art_fid.compute_art_fid_from_images(gen_imgs, ref_imgs, batch_size=2)

    del art_fid
    torch.cuda.empty_cache()

    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--methods-json", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--max-images", type=int, default=750)
    parser.add_argument("--skip-clip", action="store_true")
    parser.add_argument("--skip-musiq", action="store_true")
    parser.add_argument("--skip-artfid", action="store_true")
    args = parser.parse_args()

    with open(args.methods_json) as f:
        methods = json.load(f)

    device = torch.device(args.device)
    results = {}

    for name, cfg in methods.items():
        print(f"\n{'='*60}\n[METHOD] {name}\n{'='*60}")
        gen_dir = Path(cfg["gen_dir"])
        gen_files = collect_image_files(gen_dir, args.max_images)
        print(f"  Found {len(gen_files)} images in {gen_dir}")

        if not gen_files:
            print(f"  [SKIP] No images")
            continue

        result = {"n_images": len(gen_files)}

        if not args.skip_clip:
            clip_res = compute_clip_metrics(gen_files, device, None, args.batch_size)
            result.update({f"clip_s": clip_res["clip_s"], f"clip_t": clip_res["clip_t"]})

        lpips_res = compute_lpips(gen_files, device, args.batch_size)
        result["lpips"] = lpips_res["lpips"]

        if not args.skip_musiq:
            musiq_res = compute_musiq(gen_files, device, args.batch_size)
            result["musiq"] = musiq_res["musiq"]

        if not args.skip_artfid:
            artfid_res = compute_art_fid(gen_files, device)
            result.update(artfid_res)

        results[name] = result
        print(f"  Result: {result}")

        # Save incrementally
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)

    print(f"\n[DONE] Results saved to {args.output}")


if __name__ == "__main__":
    main()
