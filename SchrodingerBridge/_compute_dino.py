"""Compute DINO metrics (content/style/structure) for ONE (method, dataset) cell.

Handles the ACTUAL on-disk layout (flat dir, double-underscore naming) used by
results/<ds>/<method>/, unlike the original script which expected
full_eval/epoch_XXXX/images and single-underscore names.

Metrics (DINOv2-small, facebook/dinov2-small):
  - dino_content : cos(CLS(gen), CLS(content_src))   -- content preservation, higher=better
  - dino_style   : max cos(CLS(gen), CLS(style_ref)) -- style consistency,    higher=better
  - dino_structure: MSE patch-self-sim(gen, content)  -- structure preserv.,  lower=better

Usage:
  python _compute_dino.py --images_dir <results/<ds>/<method>> \
      --test_dir <datasets/.../test> --dataset {wikiart|p2a} \
      --output <state/dino/<ds>__<method>.json> [--resume] [--max_refs 30]
"""
import argparse
import json
import math
import os
import re
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image
from transformers import AutoModel

# DINOv2 preprocessing (matches DINOv2ImageProcessor) -- done offline, no HF processor download.
DINO_TRANSFORM = T.Compose([
    T.Resize(224, interpolation=Image.BICUBIC),
    T.CenterCrop(224),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

DEFAULT_WIKIART_STYLES = ["Early_Renaissance", "Impressionism", "Minimalism",
                          "Rococo", "Ukiyo_e"]
KNOWN_WIKIART_STYLES = list(DEFAULT_WIKIART_STYLES)  # will be overridden by --style_subdirs if given
KNOWN_WIKIART_STYLES_SORTED = sorted(KNOWN_WIKIART_STYLES, key=len, reverse=True)
IMG_EXTS = [".png", ".jpg", ".jpeg", ".webp", ".bmp"]


# --------------------------------------------------------------------------- #
# Filename parsing
# --------------------------------------------------------------------------- #
def parse_wikiart(stem):
    """Parse '{SrcStyle}__{artist}__to__{TgtStyle}' or '{SrcStyle}_{artist}_to_{TgtStyle}'."""
    # Try __to__ (double-underscore) format first
    if "__to__" in stem:
        left, tgt = stem.rsplit("__to__", 1)
    elif "_to_" in stem:
        left, tgt = stem.rsplit("_to_", 1)
    else:
        return None
    tgt_style = None
    for s in KNOWN_WIKIART_STYLES:
        if tgt == s or tgt.startswith(s):
            tgt_style = s
            break
    if tgt_style is None:
        return None
    src_style = None
    for s in KNOWN_WIKIART_STYLES_SORTED:
        if left == s or left.startswith(s + "_"):
            src_style = s
            break
    artist = left
    changed = True
    while changed:
        changed = False
        for s in KNOWN_WIKIART_STYLES_SORTED:
            if artist.startswith(s + "_"):
                artist = artist[len(s) + 1:]
                changed = True
            elif artist == s:
                artist = ""
                changed = True
    if not artist:
        return None
    artist = artist.lstrip("_")  # strip leading underscores from double-style prefix
    if not artist:
        return None
    # For __to__ format with double style prefix ({Style}__{Style}__{artist}),
    # strip the repeated style prefix
    if src_style and artist.startswith(src_style + "__"):
        artist = artist[len(src_style) + 2:]
    if not artist:
        return None
    return src_style, artist, tgt_style


_P2A_RE = re.compile(r"^(?P<src>[a-zA-Z]+)_(?P<id>.+?)_to_(?P<tgt>[a-zA-Z]+)$")


def parse_p2a(stem):
    """'{src_domain}_{id}_to_{tgt_domain}' e.g. cezanne_00057_to_Hayao.
    Also handles double-underscore variant '{src}__{id}__to__{tgt}' (e.g. StyTR-2 output)."""
    normalized = stem.replace("__", "_")
    m = _P2A_RE.match(normalized)
    if not m:
        return None
    return m.group("src"), m.group("id"), m.group("tgt")


# --------------------------------------------------------------------------- #
# Test-set lookup (case-insensitive folder matching)
# --------------------------------------------------------------------------- #
def _folder_map(test_dir):
    return {d.name.lower(): d for d in Path(test_dir).iterdir() if d.is_dir()}


def find_content_wikiart(test_dir, src_style, artist):
    # Test-set images carry the source-style prefix: {src_style}__{artist}.{ext}
    name = f"{src_style}__{artist}" if src_style else artist
    fmap = _folder_map(test_dir)
    if src_style and src_style.lower() in fmap:
        base = fmap[src_style.lower()]
        for ext in IMG_EXTS:
            p = base / f"{name}{ext}"
            if p.exists():
                return p
    # fallback: search every style folder
    for ext in IMG_EXTS:
        for d in Path(test_dir).iterdir():
            if d.is_dir() and (d / f"{name}{ext}").exists():
                return d / f"{name}{ext}"
    return None


def find_content_p2a(test_dir, src, id_):
    fmap = _folder_map(test_dir)
    # Try both "{src}_{id}" and "{id}" filename patterns (test dirs may use ID-only)
    candidates = [f"{src}_{id_}", id_]
    folder = fmap.get(src.lower())
    for base_name in candidates:
        if folder is not None:
            for ext in IMG_EXTS:
                p = folder / f"{base_name}{ext}"
                if p.exists():
                    return p
        # fallback across folders
        for ext in IMG_EXTS:
            for d in Path(test_dir).iterdir():
                if d.is_dir() and (d / f"{base_name}{ext}").exists():
                    return d / f"{base_name}{ext}"
    return None


def find_style_refs(test_dir, tgt, max_refs):
    fmap = _folder_map(test_dir)
    folder = fmap.get(tgt.lower())
    if folder is None:
        return []
    refs = []
    for ext in IMG_EXTS:
        refs.extend(sorted(folder.glob(f"*{ext}")))
    return refs[:max_refs]


# --------------------------------------------------------------------------- #
# DINO features
# --------------------------------------------------------------------------- #
@torch.no_grad()
def get_feats(model, transform, img, device):
    tensor = transform(img).unsqueeze(0).to(device)
    out = model(tensor, output_hidden_states=True)
    cls = out.last_hidden_state[:, 0, :]
    pen = out.hidden_states[-2][:, 1:, :]
    return cls.squeeze(0), pen.squeeze(0)


def patch_self_sim(patch_tokens):
    n = F.normalize(patch_tokens, dim=-1)
    return n @ n.T


def load_image(path):
    try:
        return Image.open(path).convert("RGB")
    except Exception:
        return None


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images_dir", required=True)
    ap.add_argument("--test_dir", required=True)
    ap.add_argument("--dataset", required=True, choices=["wikiart", "p2a"])
    ap.add_argument("--output", required=True)
    ap.add_argument("--model_name", default="facebook/dinov2-small")
    ap.add_argument("--hf_cache", default="")
    ap.add_argument("--max_refs", type=int, default=30)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--style_subdirs", default="", help="Comma-separated style names to override default wikiart styles")
    args = ap.parse_args()

    # Override known wikiart styles if --style_subdirs is given
    global KNOWN_WIKIART_STYLES, KNOWN_WIKIART_STYLES_SORTED
    if args.style_subdirs:
        KNOWN_WIKIART_STYLES = [s.strip() for s in args.style_subdirs.split(",") if s.strip()]
        KNOWN_WIKIART_STYLES_SORTED = sorted(KNOWN_WIKIART_STYLES, key=len, reverse=True)

    out_path = Path(args.output)
    if args.resume and out_path.exists():
        print(f"RESUME skip (exists): {out_path}")
        return

    images_dir = Path(args.images_dir)
    gen_files = sorted([f for f in images_dir.iterdir()
                        if f.suffix.lower() in IMG_EXTS])
    print(f"Found {len(gen_files)} generated images in {images_dir}")

    if args.hf_cache:
        os.environ["HF_HOME"] = args.hf_cache
        os.environ["TRANSFORMERS_CACHE"] = args.hf_cache
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading {args.model_name} on {device}...")
    model = AutoModel.from_pretrained(args.model_name).to(device).eval()
    processor = DINO_TRANSFORM  # offline preprocessing (no HF processor download)

    # Precompute style reference features
    style_ref_cache = {}
    print("Precomputing style references...")
    if args.dataset == "wikiart":
        for style in KNOWN_WIKIART_STYLES:
            refs = find_style_refs(args.test_dir, style, args.max_refs)
            feats = []
            for r in refs:
                img = load_image(r)
                if img is None:
                    continue
                feats.append(get_feats(model, processor, img, device))
            style_ref_cache[style] = feats
            print(f"  {style}: {len(feats)} refs")
    else:
        for dom in ["cezanne", "Hayao", "monet", "photo", "vangogh"]:
            refs = find_style_refs(args.test_dir, dom, args.max_refs)
            feats = []
            for r in refs:
                img = load_image(r)
                if img is None:
                    continue
                feats.append(get_feats(model, processor, img, device))
            style_ref_cache[dom] = feats
            print(f"  {dom}: {len(feats)} refs")

    content_cache = {}
    dc = ds_ = dst = 0.0
    n_valid = n_skip = 0

    for i, gf in enumerate(gen_files):
        stem = gf.stem
        if args.dataset == "wikiart":
            parsed = parse_wikiart(stem)
            if parsed is None:
                n_skip += 1
                continue
            src_style, artist, tgt = parsed
        else:
            parsed = parse_p2a(stem)
            if parsed is None:
                n_skip += 1
                continue
            src, id_, tgt = parsed
            src_style = artist = None

        # content source features
        if args.dataset == "wikiart":
            ckey = (src_style, artist)
            if ckey not in content_cache:
                cp = find_content_wikiart(args.test_dir, src_style, artist)
                if cp is None:
                    n_skip += 1
                    continue
                cimg = load_image(cp)
                if cimg is None:
                    n_skip += 1
                    continue
                content_cache[ckey] = get_feats(model, processor, cimg, device)
            cls_c, patch_c = content_cache[ckey]
        else:
            ckey = (src, id_)
            if ckey not in content_cache:
                cp = find_content_p2a(args.test_dir, src, id_)
                if cp is None:
                    n_skip += 1
                    continue
                cimg = load_image(cp)
                if cimg is None:
                    n_skip += 1
                    continue
                content_cache[ckey] = get_feats(model, processor, cimg, device)
            cls_c, patch_c = content_cache[ckey]

        gimg = load_image(gf)
        if gimg is None:
            n_skip += 1
            continue
        cls_g, patch_g = get_feats(model, processor, gimg, device)

        d_con = F.cosine_similarity(cls_g.unsqueeze(0), cls_c.unsqueeze(0)).item()

        refs = style_ref_cache.get(tgt, [])
        if refs:
            d_sty = max(F.cosine_similarity(cls_g.unsqueeze(0), rc.unsqueeze(0)).item()
                        for rc, _ in refs)
        else:
            d_sty = 0.0

        ss_g = patch_self_sim(patch_g)
        ss_c = patch_self_sim(patch_c)
        if ss_g.shape[0] == ss_c.shape[0]:
            d_str = F.mse_loss(ss_g, ss_c).item()
        else:
            n = min(ss_g.shape[0], ss_c.shape[0])
            d_str = F.mse_loss(ss_g[:n, :n], ss_c[:n, :n]).item()

        dc += d_con
        ds_ += d_sty
        dst += d_str
        n_valid += 1
        if (i + 1) % 100 == 0:
            print(f"  {i+1}/{len(gen_files)} valid={n_valid} skip={n_skip}")

    res = {
        "status": "ok" if n_valid > 0 else "failed",
        "dataset": args.dataset,
        "n_images": n_valid,
        "n_skipped": n_skip,
        "dino_content": dc / n_valid if n_valid else 0.0,
        "dino_style": ds_ / n_valid if n_valid else 0.0,
        "dino_structure": dst / n_valid if n_valid else 0.0,
    }
    print(f"\nDINO-con={res['dino_content']:.4f} DINO-sty={res['dino_style']:.4f} "
          f"DINO-str={res['dino_structure']:.4f} (valid={n_valid}, skip={n_skip})")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(res, indent=2, ensure_ascii=False))
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
