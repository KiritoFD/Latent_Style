"""Local evaluation of SeeDream R5-WikiArt generated images.

Reuses the exact metric definitions from scripts/_eval_unified.py
(CLIP-S = cos(gen, ref_prototype); LPIPS = gen vs src; MUSIQ).
Adapted for local machine: local dataset path, recursive glob, no remote TORCH_HOME.
"""
import argparse, json, os, re, sys, time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

DATASETS = {
    "wiki20distinct5": {
        "styles": ["Early_Renaissance", "Impressionism", "Minimalism", "Rococo", "Ukiyo_e"],
        "test_root": Path(r"G:\GitHub\Latent_Style\Dataset\wikiart_random20_512\wikiart_random20_512\images\test"),
        "image_size": 512,
    },
}


STYLE_SET = {"Early_Renaissance", "Impressionism", "Minimalism", "Rococo", "Ukiyo_e"}


def parse_filename(name: str):
    """Parse generated filename of form <src_style>__<artist>__<title>_to_<tgt_style>.png

    The source part uses '__' separators (style, artist, title); the target style
    is appended with a single '_to_' separator. The target style is one of STYLE_SET.
    """
    stem = name.rsplit(".", 1)[0] if "." in name else name
    # 1) explicit double-underscore form (legacy/other naming)
    if "__to__" in stem:
        left, tgt_style = stem.rsplit("__to__", 1)
        parts = left.split("__", 2)
        if len(parts) >= 3:
            return parts[0], parts[2], tgt_style
        elif len(parts) == 2:
            return parts[0], parts[1], tgt_style
        return None, None, None
    # 2) single '_to_' form where the target is a known style
    for style in STYLE_SET:
        suffix = "_to_" + style
        if stem.endswith(suffix):
            src_part = stem[: -len(suffix)]
            # The real source stem is of form '<style>__<artist>__<title>'.
            # The generated filename may have prepended the style name once more
            # (e.g. 'Early_Renaissance_Early_Renaissance__artist__title'); recover
            # the true stem by locating the first '<style>__' marker.
            real_src_stem = src_part
            for s in STYLE_SET:
                marker = s + "__"
                idx = src_part.find(marker)
                if idx != -1:
                    real_src_stem = src_part[idx:]
                    break
            src_style = real_src_stem.split("__")[0] if "__" in real_src_stem else None
            return src_style, real_src_stem, style
    return None, None, None


def load_image(path, size=512):
    return Image.open(path).convert("RGB").resize((size, size), Image.LANCZOS)


_CLIP_MODEL = None
_CLIP_PROCESSOR = None


def get_clip(device):
    global _CLIP_MODEL, _CLIP_PROCESSOR
    if _CLIP_MODEL is not None:
        return _CLIP_MODEL, _CLIP_PROCESSOR
    from transformers import CLIPModel, CLIPProcessor
    print("[CLIP] Loading openai/clip-vit-base-patch32 (local_files_only=True)...", flush=True)
    _CLIP_MODEL = CLIPModel.from_pretrained("openai/clip-vit-base-patch32", local_files_only=True).to(device).eval()
    _CLIP_PROCESSOR = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32", local_files_only=True)
    return _CLIP_MODEL, _CLIP_PROCESSOR


def _clip_image_features(model, inputs):
    out = model.get_image_features(**inputs)
    if isinstance(out, torch.Tensor):
        return out.float()
    if hasattr(out, "image_embeds") and out.image_embeds is not None:
        return out.image_embeds.float()
    if hasattr(out, "pooler_output") and out.pooler_output is not None:
        return out.pooler_output.float()
    if hasattr(out, "last_hidden_state") and out.last_hidden_state is not None:
        return out.last_hidden_state.mean(dim=1).float()
    raise TypeError(f"Unexpected CLIP output type: {type(out)}")


def compute_clip_s(gen_files, dataset_cfg, device, batch_size=16):
    model, processor = get_clip(device)
    styles = dataset_cfg["styles"]
    test_root = dataset_cfg["test_root"]
    img_size = dataset_cfg["image_size"]
    ref_features = {}
    for style in styles:
        style_dir = test_root / style
        if not style_dir.exists():
            print(f"[WARN] {style_dir} not found", flush=True)
            continue
        ref_files = sorted(list(style_dir.glob("*.jpg")) + list(style_dir.glob("*.png")))[:30]
        if not ref_files:
            continue
        feats = []
        for rf in ref_files:
            img = load_image(rf, img_size)
            inputs = processor(images=img, return_tensors="pt").to(device)
            with torch.no_grad():
                f = _clip_image_features(model, inputs)
                f = F.normalize(f, dim=-1)
            feats.append(f)
        proto = torch.cat(feats).mean(0, keepdim=True)
        ref_features[style] = F.normalize(proto, dim=-1)
    print("[CLIP-S] ref styles:", list(ref_features.keys()), flush=True)
    clip_s_list = []
    per_style = {s: [] for s in styles}
    for start in range(0, len(gen_files), batch_size):
        chunk = gen_files[start:start + batch_size]
        imgs = [load_image(f, img_size) for f in chunk]
        inputs = processor(images=imgs, return_tensors="pt").to(device)
        with torch.no_grad():
            gen_feats = _clip_image_features(model, inputs)
            gen_feats = F.normalize(gen_feats, dim=-1)
        for i, f in enumerate(chunk):
            _, _, tgt_style = parse_filename(f.name)
            if tgt_style and tgt_style in ref_features:
                s = float((gen_feats[i] * ref_features[tgt_style]).sum().item())
                clip_s_list.append(s)
                per_style[tgt_style].append(s)
    mean = float(np.mean(clip_s_list)) if clip_s_list else None
    per = {s: float(np.mean(v)) for s, v in per_style.items() if v}
    return mean, per


_LPIPS_MODEL = None


def get_lpips(device):
    global _LPIPS_MODEL
    if _LPIPS_MODEL is not None:
        return _LPIPS_MODEL
    import lpips
    print("[LPIPS] Loading alex...", flush=True)
    _LPIPS_MODEL = lpips.LPIPS(net="alex").to(device).eval()
    return _LPIPS_MODEL


def compute_lpips_content(gen_files, dataset_cfg, device, batch_size=8):
    lpips_fn = get_lpips(device)
    styles = dataset_cfg["styles"]
    test_root = dataset_cfg["test_root"]
    img_size = dataset_cfg["image_size"]
    src_lookup = {}
    for style in styles:
        style_dir = test_root / style
        if not style_dir.exists():
            continue
        for sf in style_dir.iterdir():
            if sf.is_file() and sf.suffix.lower() in {".jpg", ".jpeg", ".png"}:
                src_lookup[(style, sf.stem)] = sf
    lpips_list = []
    for gf in gen_files:
        parsed = parse_filename(gf.name)
        if parsed[0] is None:
            continue
        src_style, src_stem, _ = parsed
        src_file = src_lookup.get((src_style, src_stem))
        if src_file is None:
            for (s, stem), path in src_lookup.items():
                if s == src_style and src_stem in stem:
                    src_file = path
                    break
        if src_file is None:
            continue
        gen_img = load_image(gf, img_size)
        src_img = load_image(src_file, img_size)
        gen_t = torch.from_numpy(np.array(gen_img)).permute(2, 0, 1).float() / 127.5 - 1.0
        src_t = torch.from_numpy(np.array(src_img)).permute(2, 0, 1).float() / 127.5 - 1.0
        gen_t = gen_t.unsqueeze(0).to(device)
        src_t = src_t.unsqueeze(0).to(device)
        with torch.no_grad():
            d = lpips_fn(gen_t, src_t).item()
        lpips_list.append(d)
    return float(np.mean(lpips_list)) if lpips_list else None


_MUSIQ_MODEL = None


def get_musiq(device):
    global _MUSIQ_MODEL
    if _MUSIQ_MODEL is not None:
        return _MUSIQ_MODEL
    import pyiqa
    print("[MUSIQ] Loading...", flush=True)
    _MUSIQ_MODEL = pyiqa.create_metric("musiq", device=device).eval()
    return _MUSIQ_MODEL


def compute_musiq(gen_files, dataset_cfg, device, batch_size=8):
    musiq = get_musiq(device)
    from torchvision import transforms
    img_size = dataset_cfg["image_size"]
    transform = transforms.Compose([transforms.Resize((img_size, img_size)), transforms.ToTensor()])
    scores = []
    for start in range(0, len(gen_files), batch_size):
        chunk = gen_files[start:start + batch_size]
        imgs = torch.stack([transform(Image.open(f).convert("RGB")) for f in chunk], dim=0).to(device)
        with torch.no_grad():
            out = musiq(imgs)
        for v in out:
            scores.append(float(v))
    return float(np.mean(scores)) if scores else None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--image-dir", type=Path, required=True)
    p.add_argument("--dataset", default="wiki20distinct5")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--max-images", type=int, default=0)
    p.add_argument("--skip-clip", action="store_true")
    p.add_argument("--skip-lpips", action="store_true")
    p.add_argument("--skip-musiq", action="store_true")
    args = p.parse_args()

    dataset_cfg = DATASETS[args.dataset]
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")

    gen_files = sorted(list(args.image_dir.rglob("*.png")) + list(args.image_dir.rglob("*.jpg")))
    if args.max_images > 0 and len(gen_files) > args.max_images:
        gen_files = gen_files[:args.max_images]
    print(f"Found {len(gen_files)} images in {args.image_dir}", flush=True)
    if not gen_files:
        args.output.write_text(json.dumps({"error": "no images"}))
        return 1

    result = {"n_images": len(gen_files), "dataset": args.dataset, "image_dir": str(args.image_dir)}
    t0 = time.time()

    if not args.skip_clip:
        clip_s, per = compute_clip_s(gen_files, dataset_cfg, device)
        result["clip_s"] = clip_s
        result["clip_s_per_style"] = per
        print(f"CLIP-S = {clip_s}", flush=True)
        print("  per-style:", per, flush=True)

    if not args.skip_lpips:
        lpips_val = compute_lpips_content(gen_files, dataset_cfg, device)
        result["lpips"] = lpips_val
        print(f"LPIPS = {lpips_val}", flush=True)

    if not args.skip_musiq:
        musiq_val = compute_musiq(gen_files, dataset_cfg, device)
        result["musiq"] = musiq_val
        print(f"MUSIQ = {musiq_val}", flush=True)

    result["wall_seconds"] = time.time() - t0
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False))
    print(f"\n[DONE] {result}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
