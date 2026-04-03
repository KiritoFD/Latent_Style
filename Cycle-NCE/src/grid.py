from __future__ import annotations

import argparse
import csv
import importlib.util
import os
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torchvision.transforms as T
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont

try:
    import lpips
except ImportError:
    lpips = None

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from utils.inference import decode_latent, encode_image, load_vae, tensor_to_pil


@dataclass
class GridSource:
    src_style: str
    src_image_name: str
    src_stem: str
    source_path: Path


@dataclass
class GridCell:
    src_style: str
    src_image: str
    tgt_style: str
    image_path: Path | None
    clip_style: float | None
    content_lpips: float | None
    reused: bool


def _resolve(path_str: str, base: Path) -> Path:
    p = Path(path_str)
    if p.is_absolute():
        return p
    return (base / p).resolve()


def _load_eval_image_tensor(path: Path, size: int = 256) -> torch.Tensor:
    img = Image.open(path).convert("RGB").resize((size, size))
    return T.ToTensor()(img)


def _load_model_input_tensor(path: Path, size: int = 256) -> torch.Tensor:
    x = _load_eval_image_tensor(path, size=size).unsqueeze(0)
    return x * 2.0 - 1.0


def _resolve_gen_image_path(out_dir: Path, gen_image_value: str) -> Path | None:
    raw = str(gen_image_value or "").strip()
    if not raw:
        return None
    p = Path(raw)
    candidates = []
    if p.is_absolute():
        candidates.append(p)
    else:
        candidates.append((out_dir / p).resolve())
        candidates.append((out_dir / "images" / p.name).resolve())
        candidates.append((out_dir / p.name).resolve())
    for cand in candidates:
        if cand.exists():
            return cand
    return None


def _to_lpips_input(x: torch.Tensor) -> torch.Tensor:
    return x * 2.0 - 1.0


def _parse_grid_sources(grid_dir: Path, style_order: list[str]) -> list[GridSource]:
    by_style: dict[str, GridSource] = {}
    for p in sorted(grid_dir.glob("*")):
        if not p.is_file():
            continue
        stem = p.stem
        matched = None
        for style_name in sorted(style_order, key=len, reverse=True):
            prefix = f"{style_name}_"
            if stem.startswith(prefix):
                matched = style_name
                src_stem = stem[len(prefix):]
                by_style[style_name] = GridSource(
                    src_style=style_name,
                    src_image_name=f"{src_stem}{p.suffix}",
                    src_stem=src_stem,
                    source_path=p.resolve(),
                )
                break
        if matched is None:
            raise ValueError(f"Failed to infer source style from grid image name: {p.name}")
    missing = [s for s in style_order if s not in by_style]
    if missing:
        raise ValueError(f"grid_dir missing source image for styles: {missing}")
    return [by_style[s] for s in style_order]


def _read_metrics(metrics_csv: Path, reuse_dir: Path) -> dict[tuple[str, str, str], GridCell]:
    by_key: dict[tuple[str, str, str], GridCell] = {}
    if not metrics_csv.exists():
        return by_key
    with open(metrics_csv, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            src_style = str(row.get("src_style", "")).strip()
            src_image = str(row.get("src_image", "")).strip()
            tgt_style = str(row.get("tgt_style", "")).strip()
            if (not src_style) or (not src_image) or (not tgt_style):
                continue
            img_path = _resolve_gen_image_path(reuse_dir, str(row.get("gen_image", "")))
            clip_style = _to_float(row.get("clip_style"))
            content_lpips = _to_float(row.get("content_lpips"))
            by_key[(src_style, src_image, tgt_style)] = GridCell(
                src_style=src_style,
                src_image=src_image,
                tgt_style=tgt_style,
                image_path=img_path,
                clip_style=clip_style,
                content_lpips=content_lpips,
                reused=True,
            )
    return by_key


def _to_float(v: object) -> float | None:
    try:
        if v is None or str(v).strip() == "":
            return None
        return float(v)
    except Exception:
        return None


def _build_ref_prototypes(
    *,
    test_dir: Path,
    style_order: list[str],
    clip_model,
    clip_preprocess,
    device: str,
) -> dict[str, torch.Tensor]:
    protos: dict[str, torch.Tensor] = {}
    for style_name in style_order:
        s_dir = test_dir / style_name
        if not s_dir.exists():
            continue
        imgs = sorted([p for p in s_dir.iterdir() if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}])
        if not imgs:
            continue
        pils = [Image.open(p).convert("RGB") for p in imgs]
        batch = torch.stack([clip_preprocess(im) for im in pils], dim=0).to(device)
        with torch.no_grad():
            feats = clip_model.encode_image(batch).to(dtype=torch.float32)
            feats = feats / (feats.norm(dim=-1, keepdim=True) + 1e-8)
            proto = feats.mean(dim=0, keepdim=True)
            proto = proto / (proto.norm(dim=-1, keepdim=True) + 1e-8)
        protos[style_name] = proto.detach().cpu()
    return protos


def _load_builder_from_checkpoint_dir(checkpoint: Path):
    model_py = checkpoint.parent / "model.py"
    if not model_py.exists():
        from model import build_model_from_config

        return build_model_from_config

    module_name = f"grid_local_model_{abs(hash(str(model_py.resolve())))}"
    spec = importlib.util.spec_from_file_location(module_name, model_py)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load model builder from: {model_py}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    builder = getattr(module, "build_model_from_config", None)
    if builder is None:
        raise AttributeError(f"build_model_from_config not found in: {model_py}")
    return builder


def _load_checkpoint_inference(
    checkpoint: Path,
    *,
    device: str,
) -> tuple[torch.nn.Module, float, float | None]:
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    config = ckpt["config"]
    infer_cfg = config.get("inference", {})
    model_cfg = config.get("model", {})
    state_dict = ckpt["model_state_dict"]
    if any(k.startswith("_orig_mod.") for k in state_dict.keys()):
        state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}

    build_model = _load_builder_from_checkpoint_dir(checkpoint)
    model = build_model(model_cfg, use_checkpointing=False).to(device)
    try:
        model.load_state_dict(state_dict, strict=True)
    except RuntimeError:
        model.load_state_dict(state_dict, strict=False)
    model.eval()
    step_size = float(infer_cfg.get("step_size", 1.0))
    style_strength = infer_cfg.get("style_strength")
    style_strength = None if style_strength is None else float(style_strength)
    return model, step_size, style_strength


def _load_clip_openai(device: str, model_name: str, cache_dir: Path):
    import clip as openai_clip

    cache_dir.mkdir(parents=True, exist_ok=True)
    model, preprocess = openai_clip.load(model_name, device=device, download_root=str(cache_dir))
    model.eval()
    return model, preprocess


def _compute_cell_metrics(
    *,
    image_path: Path,
    source_path: Path,
    target_style: str,
    ref_prototypes: dict[str, torch.Tensor],
    clip_model,
    clip_preprocess,
    loss_fn,
    device: str,
) -> tuple[float | None, float | None]:
    clip_style = None
    content_lpips = None

    gen_img = _load_eval_image_tensor(image_path).unsqueeze(0)
    src_img = _load_eval_image_tensor(source_path).unsqueeze(0)

    if loss_fn is not None:
        with torch.no_grad():
            dist = loss_fn(
                _to_lpips_input(gen_img.to(device)),
                _to_lpips_input(src_img.to(device)),
            )
            content_lpips = float(dist.detach().cpu().view(-1)[0].item())

    tgt_proto = ref_prototypes.get(target_style)
    if tgt_proto is not None:
        pil = Image.open(image_path).convert("RGB")
        batch = torch.stack([clip_preprocess(pil)], dim=0).to(device)
        with torch.no_grad():
            feat = clip_model.encode_image(batch).to(dtype=torch.float32)
            feat = feat / (feat.norm(dim=-1, keepdim=True) + 1e-8)
            clip_style = float(F.cosine_similarity(feat.cpu(), tgt_proto, dim=-1).item())

    return clip_style, content_lpips


def _generate_missing_cells(
    *,
    checkpoint: Path,
    output_dir: Path,
    missing: list[tuple[GridSource, str]],
    style_order: list[str],
    device: str,
    image_size: int,
) -> dict[tuple[str, str, str], Path]:
    if not missing:
        return {}

    images_dir = output_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    model, step_size, style_strength = _load_checkpoint_inference(checkpoint, device=device)
    vae = load_vae(device=device)
    model_scale = float(getattr(model, "latent_scale_factor", 0.18215))
    vae_scale = float(getattr(getattr(vae, "config", None), "scaling_factor", model_scale))
    scale_in = model_scale / max(vae_scale, 1e-8)
    scale_out = vae_scale / max(model_scale, 1e-8)

    saved: dict[tuple[str, str, str], Path] = {}
    for src, tgt_style in missing:
        target_style_id = style_order.index(tgt_style)
        x = _load_model_input_tensor(src.source_path, size=image_size)
        z = encode_image(vae, x, device=device)
        if abs(scale_in - 1.0) > 1e-4:
            z = z * scale_in
        z = z.to(device=device, dtype=next(model.parameters()).dtype)
        with torch.no_grad():
            z_out = model.integrate(
                z,
                style_id=target_style_id,
                num_steps=1,
                step_size=step_size,
                style_strength=style_strength,
            )
        if abs(scale_out - 1.0) > 1e-4:
            z_out = z_out * scale_out
        out = decode_latent(vae, z_out, device=device)
        dst = images_dir / f"{src.src_style}_{src.src_stem}_to_{tgt_style}.jpg"
        tensor_to_pil(out).save(dst)
        saved[(src.src_style, src.src_image_name, tgt_style)] = dst
    return saved


def _save_grid(
    *,
    rows: list[GridSource],
    style_order: list[str],
    cells: dict[tuple[str, str, str], GridCell],
    output_path: Path,
) -> None:
    existing = [c.image_path for c in cells.values() if c.image_path is not None and c.image_path.exists()]
    if not existing:
        raise RuntimeError("No grid images available to render.")

    sizes = []
    for p in existing:
        with Image.open(p) as im:
            sizes.append(im.size)
    cell_w = max(w for w, _ in sizes)
    cell_h = max(h for _, h in sizes)

    try:
        font = ImageFont.truetype("arial.ttf", size=28)
        font_small = ImageFont.truetype("arial.ttf", size=16)
    except Exception:
        font = ImageFont.load_default()
        font_small = font

    pad = 18
    header_h = 56
    metric_h = 24
    left_w = 280
    bg = (0, 0, 0)
    fg = (255, 255, 255)

    canvas_w = left_w + len(style_order) * cell_w + (len(style_order) + 1) * pad
    canvas_h = header_h + len(rows) * (cell_h + metric_h) + (len(rows) + 1) * pad
    canvas = Image.new("RGB", (canvas_w, canvas_h), color=bg)
    draw = ImageDraw.Draw(canvas)

    for ci, tgt_style in enumerate(style_order):
        x = left_w + pad + ci * (cell_w + pad)
        draw.text((x, 8), tgt_style, fill=fg, font=font)

    for ri, src in enumerate(rows):
        y = header_h + pad + ri * (cell_h + metric_h + pad)
        draw.text((6, y + max(0, (cell_h - 28) // 2)), src.src_style, fill=fg, font=font)
        draw.text((6, y + max(0, (cell_h - 28) // 2) + 30), Path(src.src_image_name).stem, fill=(200, 200, 200), font=font_small)
        for ci, tgt_style in enumerate(style_order):
            px = left_w + pad + ci * (cell_w + pad)
            py = y
            cell = cells.get((src.src_style, src.src_image_name, tgt_style))
            if cell is None or cell.image_path is None or (not cell.image_path.exists()):
                draw.rectangle((px, py, px + cell_w, py + cell_h), outline=(90, 90, 90), width=2)
                draw.text((px + 8, py + 8), "MISSING", fill=(180, 120, 120), font=font_small)
                continue
            with Image.open(cell.image_path).convert("RGB") as im:
                canvas.paste(im, (px, py))
            clip_text = "NA" if cell.clip_style is None else f"{cell.clip_style:.3f}"
            lpips_text = "NA" if cell.content_lpips is None else f"{cell.content_lpips:.3f}"
            draw.text((px + 4, py + cell_h + 3), f"clip={clip_text} lpips={lpips_text}", fill=(230, 230, 230), font=font_small)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path, format="PNG")


def _write_metrics_csv(
    *,
    rows: list[GridSource],
    style_order: list[str],
    cells: dict[tuple[str, str, str], GridCell],
    output_csv: Path,
) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(output_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["src_style", "tgt_style", "src_image", "gen_image", "clip_style", "content_lpips", "reused"],
        )
        writer.writeheader()
        for src in rows:
            for tgt_style in style_order:
                cell = cells.get((src.src_style, src.src_image_name, tgt_style))
                writer.writerow(
                    {
                        "src_style": src.src_style,
                        "tgt_style": tgt_style,
                        "src_image": src.src_image_name,
                        "gen_image": str(cell.image_path) if cell and cell.image_path else "",
                        "clip_style": "" if cell is None or cell.clip_style is None else cell.clip_style,
                        "content_lpips": "" if cell is None or cell.content_lpips is None else cell.content_lpips,
                        "reused": bool(cell.reused) if cell is not None else False,
                    }
                )


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a fixed 5x5 grid from grid5 images, reusing full_eval results when possible.")
    parser.add_argument("--grid_dir", type=str, default="../../style_data/grid5")
    parser.add_argument("--reuse_dir", type=str, required=True, help="Existing full_eval directory containing metrics.csv and generated images")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--style_subdirs", type=str, default="photo,Hayao,monet,vangogh,cezanne")
    parser.add_argument("--test_dir", type=str, default="../../style_data/overfit50")
    parser.add_argument("--checkpoint", type=str, default="", help="Optional Cycle-NCE checkpoint used to generate missing cells")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--clip_openai_model", type=str, default="ViT-B/32")
    parser.add_argument("--clip_cache_dir", type=str, default="../eval_cache/clip_openai")
    args = parser.parse_args()

    src_dir = Path(__file__).resolve().parent
    grid_dir = _resolve(args.grid_dir, src_dir)
    reuse_dir = _resolve(args.reuse_dir, src_dir)
    output_dir = _resolve(args.output_dir, src_dir)
    test_dir = _resolve(args.test_dir, src_dir)
    clip_cache_dir = _resolve(args.clip_cache_dir, src_dir)
    checkpoint = _resolve(args.checkpoint, src_dir) if str(args.checkpoint).strip() else None
    style_order = [x.strip() for x in str(args.style_subdirs).split(",") if x.strip()]

    if not grid_dir.exists():
        raise FileNotFoundError(f"grid_dir not found: {grid_dir}")
    if not reuse_dir.exists():
        raise FileNotFoundError(f"reuse_dir not found: {reuse_dir}")
    if not test_dir.exists():
        raise FileNotFoundError(f"test_dir not found: {test_dir}")

    device = args.device
    if device.startswith("cuda") and not torch.cuda.is_available():
        device = "cpu"

    rows = _parse_grid_sources(grid_dir, style_order)
    metrics_map = _read_metrics(reuse_dir / "metrics.csv", reuse_dir)

    cells: dict[tuple[str, str, str], GridCell] = {}
    missing: list[tuple[GridSource, str]] = []
    for src in rows:
        for tgt_style in style_order:
            key = (src.src_style, src.src_image_name, tgt_style)
            cell = metrics_map.get(key)
            if cell is None or cell.image_path is None or (not cell.image_path.exists()):
                missing.append((src, tgt_style))
                cells[key] = GridCell(
                    src_style=src.src_style,
                    src_image=src.src_image_name,
                    tgt_style=tgt_style,
                    image_path=None,
                    clip_style=cell.clip_style if cell else None,
                    content_lpips=cell.content_lpips if cell else None,
                    reused=False,
                )
            else:
                cells[key] = cell

    print(f"[reuse] found {len(cells) - len(missing)}/{len(cells)} cells directly from {reuse_dir}")

    if missing and checkpoint is not None:
        print(f"[infer] generating {len(missing)} missing cells from checkpoint: {checkpoint}")
        generated = _generate_missing_cells(
            checkpoint=checkpoint,
            output_dir=output_dir,
            missing=missing,
            style_order=style_order,
            device=device,
            image_size=int(args.image_size),
        )
        for key, p in generated.items():
            cells[key].image_path = p
            cells[key].reused = False
    elif missing:
        print(f"[warn] {len(missing)} cells missing and no checkpoint provided; they will be rendered as MISSING")

    need_metric_compute = False
    for src in rows:
        for tgt_style in style_order:
            cell = cells[(src.src_style, src.src_image_name, tgt_style)]
            if cell.image_path is None or (not cell.image_path.exists()):
                continue
            if cell.clip_style is None or cell.content_lpips is None:
                need_metric_compute = True
                break
        if need_metric_compute:
            break

    if need_metric_compute:
        if lpips is None:
            raise RuntimeError("lpips package is required for grid metrics.")
        loss_fn = lpips.LPIPS(net="vgg", verbose=False).to(device)
        clip_model, clip_preprocess = _load_clip_openai(device=device, model_name=args.clip_openai_model, cache_dir=clip_cache_dir)
        ref_prototypes = _build_ref_prototypes(
            test_dir=test_dir,
            style_order=style_order,
            clip_model=clip_model,
            clip_preprocess=clip_preprocess,
            device=device,
        )

        for src in rows:
            for tgt_style in style_order:
                key = (src.src_style, src.src_image_name, tgt_style)
                cell = cells[key]
                if cell.image_path is None or (not cell.image_path.exists()):
                    continue
                if cell.clip_style is not None and cell.content_lpips is not None:
                    continue
                clip_style, content_lpips = _compute_cell_metrics(
                    image_path=cell.image_path,
                    source_path=src.source_path,
                    target_style=tgt_style,
                    ref_prototypes=ref_prototypes,
                    clip_model=clip_model,
                    clip_preprocess=clip_preprocess,
                    loss_fn=loss_fn,
                    device=device,
                )
                cell.clip_style = clip_style
                cell.content_lpips = content_lpips

    _write_metrics_csv(
        rows=rows,
        style_order=style_order,
        cells=cells,
        output_csv=output_dir / "grid_metrics.csv",
    )
    _save_grid(
        rows=rows,
        style_order=style_order,
        cells=cells,
        output_path=output_dir / "grid.png",
    )
    print(f"[done] grid image: {output_dir / 'grid.png'}")
    print(f"[done] grid metrics: {output_dir / 'grid_metrics.csv'}")


if __name__ == "__main__":
    main()
