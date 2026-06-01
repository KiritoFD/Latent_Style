from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont, ImageOps
from tqdm.auto import tqdm


def _repo_src_path() -> Path:
    return Path(__file__).resolve().parents[1] / "src"


SRC_PATH = str(_repo_src_path())
if SRC_PATH not in sys.path:
    sys.path.insert(0, SRC_PATH)

from utils.inference import LGTInference, decode_latent, load_vae  # noqa: E402


IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".webp", ".bmp")


def _load_checkpoint_config(checkpoint: Path) -> dict:
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    cfg = payload.get("config", {})
    if not isinstance(cfg, dict):
        raise ValueError(f"Checkpoint config is not a dict: {checkpoint}")
    return cfg


def _resolve_classes(checkpoint: Path, raw: str | None, latent_root: Path) -> list[str]:
    if raw:
        return [item.strip() for item in raw.split(",") if item.strip()]
    cfg = _load_checkpoint_config(checkpoint)
    data = cfg.get("data", {}) if isinstance(cfg.get("data", {}), dict) else {}
    style_subdirs = data.get("style_subdirs")
    if isinstance(style_subdirs, list) and style_subdirs:
        return [str(item) for item in style_subdirs]
    return [p.name for p in sorted(latent_root.iterdir(), key=lambda x: x.name) if p.is_dir()]


def _image_for_latent(image_dir: Path, latent_path: Path) -> Path:
    stem = latent_path.stem
    for ext in IMAGE_EXTS:
        candidate = image_dir / f"{stem}{ext}"
        if candidate.exists():
            return candidate
    matches = [p for p in image_dir.iterdir() if p.is_file() and p.stem == stem]
    if matches:
        return sorted(matches, key=lambda p: p.name)[0]
    raise FileNotFoundError(f"No source image for latent: {latent_path}")


def _load_latent(path: Path) -> torch.Tensor:
    latent = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(latent, dict):
        for key in ("latent", "z", "tensor"):
            if key in latent:
                latent = latent[key]
                break
    if not torch.is_tensor(latent):
        raise TypeError(f"Unsupported latent payload: {path}")
    latent = latent.float()
    if latent.ndim == 4 and latent.shape[0] == 1:
        latent = latent[0]
    if latent.ndim != 3:
        raise ValueError(f"Expected [C,H,W] latent, got {tuple(latent.shape)} from {path}")
    return latent.contiguous()


def _load_metric_tensor(path: Path, image_size: int) -> torch.Tensor:
    with Image.open(path) as image:
        image = ImageOps.exif_transpose(image).convert("RGB")
        image = ImageOps.fit(image, (image_size, image_size), method=Image.Resampling.LANCZOS, centering=(0.5, 0.5))
        arr = np.asarray(image, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(arr).permute(2, 0, 1)
    return tensor * 2.0 - 1.0


def _load_clip(args: argparse.Namespace, device: torch.device):
    from transformers import CLIPModel, CLIPProcessor

    model_name = str(args.clip_model).strip()
    source = Path(model_name)
    if source.exists():
        source_arg = str(source.resolve())
    else:
        source_arg = model_name
    kwargs = {
        "cache_dir": str(args.hf_cache_dir),
        "local_files_only": not bool(args.clip_allow_network),
    }
    model = CLIPModel.from_pretrained(source_arg, **kwargs).to(device)
    processor = CLIPProcessor.from_pretrained(source_arg, **kwargs)
    model.eval()

    def extract_image_features(output):
        if torch.is_tensor(output):
            return output
        for attr in ("image_embeds", "pooler_output", "text_embeds"):
            value = getattr(output, attr, None)
            if torch.is_tensor(value):
                return value
        if isinstance(output, dict):
            for key in ("image_embeds", "pooler_output", "text_embeds", "last_hidden_state"):
                value = output.get(key)
                if torch.is_tensor(value):
                    if key == "last_hidden_state" and value.ndim == 3:
                        return value[:, 0]
                    return value
        if isinstance(output, (tuple, list)):
            for value in output:
                if torch.is_tensor(value):
                    if value.ndim == 3:
                        return value[:, 0]
                    return value
        raise TypeError(f"Unsupported CLIP image feature output: {type(output)!r}")

    @torch.no_grad()
    def encode_pils(images: list[Image.Image]) -> torch.Tensor:
        inputs = processor(images=images, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        feats = extract_image_features(model.get_image_features(**inputs)).float()
        return feats / feats.norm(dim=-1, keepdim=True).clamp_min(1e-8)

    return encode_pils


def _pil_from_tensor_01(tensor: torch.Tensor) -> Image.Image:
    t = tensor.detach().cpu().float().clamp(0.0, 1.0)
    arr = (t.permute(1, 2, 0).numpy() * 255.0).round().astype(np.uint8)
    return Image.fromarray(arr)


def _save_png(tensor_01: torch.Tensor, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    _pil_from_tensor_01(tensor_01).save(path)


def _build_items(latent_root: Path, image_root: Path, classes: list[str], max_per_source_style: int = 0) -> list[dict]:
    items: list[dict] = []
    for style_id, style_name in enumerate(classes):
        latent_dir = latent_root / style_name
        image_dir = image_root / style_name
        if not latent_dir.is_dir():
            raise FileNotFoundError(f"Missing latent dir: {latent_dir}")
        if not image_dir.is_dir():
            raise FileNotFoundError(f"Missing image dir: {image_dir}")
        latent_paths = sorted(latent_dir.glob("*.pt"), key=lambda p: p.name)
        if max_per_source_style > 0:
            latent_paths = latent_paths[:max_per_source_style]
        for latent_path in latent_paths:
            image_path = _image_for_latent(image_dir, latent_path)
            items.append(
                {
                    "source_style_id": style_id,
                    "source_style": style_name,
                    "latent_path": latent_path,
                    "image_path": image_path,
                    "stem": latent_path.stem,
                }
            )
    return items


@torch.no_grad()
def _compute_style_prototypes(
    *,
    image_root: Path,
    classes: list[str],
    encode_clip,
    batch_size: int,
    device: torch.device,
) -> dict[int, torch.Tensor]:
    prototypes: dict[int, torch.Tensor] = {}
    for style_id, style_name in enumerate(classes):
        image_paths = sorted(
            [p for p in (image_root / style_name).iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS],
            key=lambda p: p.name,
        )
        feats = []
        for start in tqdm(range(0, len(image_paths), batch_size), desc=f"clip refs {style_name}"):
            paths = image_paths[start : start + batch_size]
            pils = [Image.open(p).convert("RGB") for p in paths]
            try:
                feats.append(encode_clip(pils).detach())
            finally:
                for pil in pils:
                    pil.close()
        if not feats:
            raise RuntimeError(f"No reference images for style {style_name}")
        matrix = torch.cat(feats, dim=0)
        proto = matrix.mean(dim=0, keepdim=True)
        prototypes[style_id] = (proto / proto.norm(dim=-1, keepdim=True).clamp_min(1e-8)).to(device)
    return prototypes


def _row_summary(rows: list[dict]) -> dict:
    metric_keys = [
        "clip_style",
        "source_to_target_clip",
        "clip_style_gain",
        "clip_dir",
        "clip_content",
        "content_lpips",
        "latent_delta_l2",
        "latent_delta_abs_mean",
    ]

    def mean(key: str, pool: list[dict]) -> float:
        vals = [float(r[key]) for r in pool]
        return float(np.mean(vals)) if vals else 0.0

    cross_rows = [r for r in rows if r["src_style"] != r["tgt_style"]]
    identity_rows = [r for r in rows if r["src_style"] == r["tgt_style"]]
    by_target = {}
    for target in sorted({r["tgt_style"] for r in rows}):
        pool = [r for r in rows if r["tgt_style"] == target]
        by_target[target] = {
            "count": len(pool),
            **{key: mean(key, pool) for key in metric_keys},
        }

    by_pair = {}
    for src in sorted({r["src_style"] for r in rows}):
        for tgt in sorted({r["tgt_style"] for r in rows}):
            pool = [r for r in rows if r["src_style"] == src and r["tgt_style"] == tgt]
            if pool:
                by_pair[f"{src}->{tgt}"] = {
                    "count": len(pool),
                    **{key: mean(key, pool) for key in metric_keys},
                }

    return {
        "count": len(rows),
        "overall": {
            **{key: mean(key, rows) for key in metric_keys},
        },
        "cross_only": {
            "count": len(cross_rows),
            **{key: mean(key, cross_rows) for key in metric_keys},
        },
        "identity_only": {
            "count": len(identity_rows),
            **{key: mean(key, identity_rows) for key in metric_keys},
        },
        "by_target": by_target,
        "by_pair": by_pair,
    }


def _make_grid(rows: list[dict], image_root: Path, classes: list[str], out_path: Path, cell: int = 192) -> None:
    first_stem_by_src = {}
    for row in rows:
        first_stem_by_src.setdefault(row["src_style"], Path(row["src_image"]).stem)
    index = {(r["src_style"], r["tgt_style"], Path(r["src_image"]).stem): r for r in rows}

    label_h = 30
    cols = len(classes) + 1
    rows_n = len(classes)
    canvas = Image.new("RGB", (cols * cell, rows_n * (cell + label_h)), (20, 20, 20))
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype("arial.ttf", 13)
    except Exception:
        font = ImageFont.load_default()

    for r_idx, src_style in enumerate(classes):
        y = r_idx * (cell + label_h)
        stem = first_stem_by_src.get(src_style)
        src_path = None
        if stem:
            for ext in IMAGE_EXTS:
                candidate = image_root / src_style / f"{stem}{ext}"
                if candidate.exists():
                    src_path = candidate
                    break
        if src_path is not None:
            with Image.open(src_path) as im:
                im = ImageOps.exif_transpose(im).convert("RGB")
                im = ImageOps.fit(im, (cell, cell), method=Image.Resampling.LANCZOS)
                canvas.paste(im, (0, y + label_h))
        draw.text((4, y + 6), f"{src_style} src", fill=(240, 240, 240), font=font)
        for c_idx, tgt_style in enumerate(classes, start=1):
            row = index.get((src_style, tgt_style, stem))
            x = c_idx * cell
            if row:
                with Image.open(row["gen_image"]) as im:
                    im = im.convert("RGB").resize((cell, cell), Image.Resampling.LANCZOS)
                    canvas.paste(im, (x, y + label_h))
                draw.text(
                    (x + 4, y + 6),
                    f"{tgt_style} c={float(row['clip_style']):.3f} l={float(row['content_lpips']):.3f}",
                    fill=(240, 240, 240),
                    font=font,
                )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path)


def evaluate(args: argparse.Namespace) -> dict:
    torch.set_float32_matmul_precision("high")
    torch.backends.cudnn.benchmark = True
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    checkpoint = Path(args.checkpoint)
    latent_root = Path(args.latent_root)
    image_root = Path(args.image_root)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    gen_dir = out_dir / "generated"
    gen_dir.mkdir(parents=True, exist_ok=True)

    classes = _resolve_classes(checkpoint, args.classes, latent_root)
    items = _build_items(latent_root, image_root, classes, max_per_source_style=int(args.max_per_source_style))
    expected = len(classes) * len(items)
    print(f"classes={classes}")
    print(f"source_items={len(items)} transfers={expected}")

    encode_clip = _load_clip(args, device)
    style_prototypes = _compute_style_prototypes(
        image_root=image_root,
        classes=classes,
        encode_clip=encode_clip,
        batch_size=max(1, int(args.ref_batch_size)),
        device=device,
    )

    import lpips

    lpips_model = lpips.LPIPS(net=args.lpips_net, verbose=False).to(device).eval()
    vae = load_vae(device=str(device), model_id=args.vae_model, cache_dir=str(args.hf_cache_dir))
    infer = LGTInference(
        str(checkpoint),
        device=str(device),
        num_steps=int(args.num_steps),
        step_size=float(args.step_size),
        style_strength=float(args.style_strength),
    )

    source_clip_cache: dict[str, torch.Tensor] = {}
    source_lpips_cache: dict[str, torch.Tensor] = {}
    rows: list[dict] = []
    metric_batch = max(1, int(args.batch_size))

    target_chunk = max(1, min(len(classes), int(args.target_chunk_size)))
    default_decode_bs = max(1, metric_batch * target_chunk)
    vae_decode_bs = max(1, int(args.vae_decode_batch_size) if int(args.vae_decode_batch_size) > 0 else default_decode_bs)

    for target_start in range(0, len(classes), target_chunk):
        target_ids_chunk = list(range(target_start, min(len(classes), target_start + target_chunk)))
        target_label = ",".join(classes[idx] for idx in target_ids_chunk)
        for start in tqdm(range(0, len(items), metric_batch), desc=f"generate -> {target_label}"):
            batch = items[start : start + metric_batch]
            latents = torch.stack([_load_latent(item["latent_path"]) for item in batch], dim=0).to(device)
            with torch.no_grad():
                repeated_latents = latents.repeat(len(target_ids_chunk), 1, 1, 1)
                target_ids = torch.cat(
                    [
                        torch.full((latents.shape[0],), target_id, dtype=torch.long, device=device)
                        for target_id in target_ids_chunk
                    ],
                    dim=0,
                )
                out_latents = infer.transfer_style(repeated_latents, target_style_id=target_ids, num_steps=int(args.num_steps))
                latent_delta = (out_latents.float() - repeated_latents.float()).flatten(1)
                latent_delta_l2 = latent_delta.norm(dim=1).detach().cpu().numpy()
                latent_delta_abs_mean = latent_delta.abs().mean(dim=1).detach().cpu().numpy()
            decoded_parts = []
            for dec_start in range(0, out_latents.shape[0], vae_decode_bs):
                dec_end = min(out_latents.shape[0], dec_start + vae_decode_bs)
                with torch.no_grad():
                    decoded_parts.append(decode_latent(vae, out_latents[dec_start:dec_end], device=str(device)).detach().cpu())
            decoded = torch.cat(decoded_parts, dim=0)

            for local_target_idx, target_id in enumerate(target_ids_chunk):
                target_name = classes[target_id]
                offset = local_target_idx * len(batch)
                decoded_slice = decoded[offset : offset + len(batch)]
                delta_l2_slice = latent_delta_l2[offset : offset + len(batch)]
                delta_abs_slice = latent_delta_abs_mean[offset : offset + len(batch)]

                gen_pils = []
                gen_lpips = []
                src_lpips = []
                for i, item in enumerate(batch):
                    gen_name = f"{item['source_style']}__{item['stem']}__to__{target_name}.png"
                    gen_path = gen_dir / gen_name
                    _save_png(decoded_slice[i], gen_path)
                    gen_pils.append(_pil_from_tensor_01(decoded_slice[i]))
                    gen_lpips.append(decoded_slice[i] * 2.0 - 1.0)

                    src_key = str(item["image_path"].resolve())
                    if src_key not in source_lpips_cache:
                        source_lpips_cache[src_key] = _load_metric_tensor(item["image_path"], int(args.image_size))
                    src_lpips.append(source_lpips_cache[src_key])
                    if src_key not in source_clip_cache:
                        with Image.open(item["image_path"]) as im:
                            pil = ImageOps.exif_transpose(im).convert("RGB")
                            source_clip_cache[src_key] = encode_clip([pil]).detach().cpu()[0]

                with torch.no_grad():
                    gen_clip = encode_clip(gen_pils)
                    src_clip = torch.stack([source_clip_cache[str(item["image_path"].resolve())] for item in batch], dim=0).to(device)
                    target_proto = style_prototypes[target_id].expand(gen_clip.shape[0], -1)
                    clip_style = F.cosine_similarity(gen_clip, target_proto, dim=-1).detach().cpu().numpy()
                    clip_content = F.cosine_similarity(gen_clip, src_clip, dim=-1).detach().cpu().numpy()
                    source_to_target_clip = F.cosine_similarity(src_clip, target_proto, dim=-1)
                    clip_style_gain = F.cosine_similarity(gen_clip, target_proto, dim=-1) - source_to_target_clip
                    dir_gen = gen_clip - src_clip
                    dir_tgt = target_proto - src_clip
                    dir_gen = dir_gen / dir_gen.norm(dim=-1, keepdim=True).clamp_min(1e-8)
                    dir_tgt = dir_tgt / dir_tgt.norm(dim=-1, keepdim=True).clamp_min(1e-8)
                    clip_dir = F.cosine_similarity(dir_gen, dir_tgt, dim=-1).detach().cpu().numpy()
                    source_to_target_clip = source_to_target_clip.detach().cpu().numpy()
                    clip_style_gain = clip_style_gain.detach().cpu().numpy()
                    lp_gen = torch.stack(gen_lpips, dim=0).to(device)
                    lp_src = torch.stack(src_lpips, dim=0).to(device)
                    content_lpips = lpips_model(lp_gen, lp_src).view(-1).detach().cpu().numpy()

                for i, item in enumerate(batch):
                    gen_name = f"{item['source_style']}__{item['stem']}__to__{target_name}.png"
                    rows.append(
                        {
                            "src_style": item["source_style"],
                            "tgt_style": target_name,
                            "src_image": str(item["image_path"]),
                            "src_latent": str(item["latent_path"]),
                            "gen_image": str(gen_dir / gen_name),
                            "clip_style": float(clip_style[i]),
                            "source_to_target_clip": float(source_to_target_clip[i]),
                            "clip_style_gain": float(clip_style_gain[i]),
                            "clip_dir": float(clip_dir[i]),
                            "clip_content": float(clip_content[i]),
                            "content_lpips": float(content_lpips[i]),
                            "latent_delta_l2": float(delta_l2_slice[i]),
                            "latent_delta_abs_mean": float(delta_abs_slice[i]),
                        }
                    )

                for pil in gen_pils:
                    pil.close()

    csv_path = out_dir / "metrics.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "src_style",
                "tgt_style",
                "src_image",
                "src_latent",
                "gen_image",
                "clip_style",
                "source_to_target_clip",
                "clip_style_gain",
                "clip_dir",
                "clip_content",
                "content_lpips",
                "latent_delta_l2",
                "latent_delta_abs_mean",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    summary = _row_summary(rows)
    summary.update(
        {
            "checkpoint": str(checkpoint),
            "latent_root": str(latent_root),
            "image_root": str(image_root),
            "classes": classes,
            "num_steps": int(args.num_steps),
            "step_size": float(args.step_size),
            "style_strength": float(args.style_strength),
        }
    )
    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    _make_grid(rows, image_root, classes, out_dir / "grid_first_per_class.png")
    print(json.dumps(summary["overall"], indent=2))
    print(f"wrote {csv_path}")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate 512 WikiArt latent LANCET with original test images.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--latent-root", required=True)
    parser.add_argument("--image-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--classes", default="")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--target-chunk-size", type=int, default=1)
    parser.add_argument("--vae-decode-batch-size", type=int, default=0)
    parser.add_argument("--ref-batch-size", type=int, default=16)
    parser.add_argument("--max-per-source-style", type=int, default=0)
    parser.add_argument("--num-steps", type=int, default=4)
    parser.add_argument("--step-size", type=float, default=1.0)
    parser.add_argument("--style-strength", type=float, default=1.0)
    parser.add_argument("--vae-model", default="ema")
    parser.add_argument("--hf-cache-dir", default="I:/Github/Latent_Style/eval_cache/hf")
    parser.add_argument(
        "--clip-model",
        default="I:/Github/Latent_Style/eval_cache/manual_clip/openai-clip-vit-base-patch32",
    )
    parser.add_argument("--clip-allow-network", action="store_true")
    parser.add_argument("--lpips-net", default="vgg", choices=["vgg", "alex", "squeeze"])
    return parser.parse_args()


def main() -> None:
    evaluate(parse_args())


if __name__ == "__main__":
    main()
