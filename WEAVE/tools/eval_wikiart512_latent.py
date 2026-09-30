from __future__ import annotations

import argparse
import concurrent.futures
import csv
import json
import sys
import time
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

from utils.inference import LGTInference, ORTVAEDecoder, decode_latent, load_vae  # noqa: E402


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


def _load_image_tensor_01(path: Path, image_size: int) -> torch.Tensor:
    return (_load_metric_tensor(path, image_size) + 1.0) * 0.5


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


def _load_clip_tensor(args: argparse.Namespace, device: torch.device):
    from transformers import CLIPModel

    model_name = str(args.clip_model).strip()
    source = Path(model_name)
    source_arg = str(source.resolve()) if source.exists() else model_name
    kwargs = {
        "cache_dir": str(args.hf_cache_dir),
        "local_files_only": not bool(args.clip_allow_network),
    }
    model = CLIPModel.from_pretrained(source_arg, **kwargs).to(device)
    model.eval()
    mean = torch.tensor((0.48145466, 0.4578275, 0.40821073), device=device).view(1, 3, 1, 1)
    std = torch.tensor((0.26862954, 0.26130258, 0.27577711), device=device).view(1, 3, 1, 1)

    @torch.no_grad()
    def encode_tensors(images_01: torch.Tensor) -> torch.Tensor:
        x = images_01.to(device=device, dtype=torch.float32).clamp(0.0, 1.0)
        if x.shape[-2:] != (224, 224):
            x = F.interpolate(x, size=(224, 224), mode="bicubic", align_corners=False, antialias=True)
        x = (x - mean) / std
        feats = model.get_image_features(pixel_values=x).float()
        return feats / feats.norm(dim=-1, keepdim=True).clamp_min(1e-8)

    return encode_tensors


def _pil_from_tensor_01(tensor: torch.Tensor) -> Image.Image:
    t = tensor.detach().cpu().float().clamp(0.0, 1.0)
    arr = (t.permute(1, 2, 0).numpy() * 255.0).round().astype(np.uint8)
    return Image.fromarray(arr)


def _save_png(tensor_01: torch.Tensor, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    _pil_from_tensor_01(tensor_01).save(path)


def _sync_if_needed(device: torch.device, enabled: bool) -> None:
    if enabled and device.type == "cuda":
        torch.cuda.synchronize(device)


def _add_elapsed(timers: dict[str, float], key: str, start: float) -> None:
    timers[key] = timers.get(key, 0.0) + (time.perf_counter() - start)


def _effective_rank(matrix: torch.Tensor) -> tuple[float, list[float]]:
    _, svals, _ = torch.linalg.svd(matrix - matrix.mean(dim=0, keepdim=True), full_matrices=False)
    rank = float((svals.sum().square() / svals.square().sum().clamp_min(1e-8)).item())
    return rank, [float(v) for v in svals.tolist()]


def _delta_diagnostics(delta_sums: dict[int, torch.Tensor], delta_counts: dict[int, int], classes: list[str]) -> dict:
    active = [idx for idx in range(len(classes)) if delta_counts.get(idx, 0) > 0 and idx in delta_sums]
    if len(active) < 2:
        return {}
    means = []
    target_rows = {}
    for idx in active:
        mean_vec = delta_sums[idx] / max(1, delta_counts[idx])
        means.append(mean_vec)
        target_rows[classes[idx]] = {
            "count": int(delta_counts[idx]),
            "delta_mean_l2": float(mean_vec.norm().item()),
        }
    matrix = torch.stack(means, dim=0)
    rank, svals = _effective_rank(matrix)
    gram = F.normalize(matrix, dim=1) @ F.normalize(matrix, dim=1).T
    pair_rows = {}
    for local_i, idx_a in enumerate(active):
        for local_j, idx_b in enumerate(active[local_i + 1 :], start=local_i + 1):
            va = matrix[local_i]
            vb = matrix[local_j]
            pair_rows[f"{classes[idx_a]}->{classes[idx_b]}"] = {
                "delta_mean_l2": float((va - vb).norm().item()),
                "delta_mean_cos": float(gram[local_i, local_j].item()),
            }
    return {
        "generated_delta_effective_rank": rank,
        "generated_delta_rank_svals": svals,
        "generated_delta_mean_offdiag_cos": float(((gram.sum() - torch.diagonal(gram).sum()) / max(1, len(active) * (len(active) - 1))).item()),
        "generated_delta_by_target": target_rows,
        "generated_delta_by_pair": pair_rows,
    }


def _delta_variance_decomposition(
    delta_vectors: list[torch.Tensor],
    *,
    source_style_ids: list[int],
    target_style_ids: list[int],
    source_image_keys: list[str],
    classes: list[str],
) -> dict:
    if not delta_vectors:
        return {}
    matrix = torch.stack([x.float() for x in delta_vectors], dim=0)
    if matrix.shape[0] < 2:
        return {}
    grand = matrix.mean(dim=0, keepdim=True)
    centered = matrix - grand
    total_ss = centered.square().sum().clamp_min(1e-8)

    def _between_ratio(labels: list[object], values: torch.Tensor = matrix) -> tuple[float, int]:
        groups: dict[object, list[int]] = {}
        for idx, label in enumerate(labels):
            groups.setdefault(label, []).append(idx)
        if len(groups) < 2:
            return 0.0, len(groups)
        base = values.mean(dim=0, keepdim=True)
        ss = values.new_tensor(0.0)
        denom = (values - base).square().sum().clamp_min(1e-8)
        for idxs in groups.values():
            idx_t = torch.as_tensor(idxs, dtype=torch.long, device=values.device)
            mean = values.index_select(0, idx_t).mean(dim=0, keepdim=True)
            ss = ss + len(idxs) * (mean - base).square().sum()
        return float((ss / denom).item()), len(groups)

    target_ratio, target_groups = _between_ratio(target_style_ids)
    source_style_ratio, source_style_groups = _between_ratio(source_style_ids)
    source_image_ratio, source_image_groups = _between_ratio(source_image_keys)
    pair_labels = [f"{s}->{t}" for s, t in zip(source_style_ids, target_style_ids, strict=False)]
    pair_ratio, pair_groups = _between_ratio(pair_labels)

    image_groups: dict[str, list[int]] = {}
    for idx, key in enumerate(source_image_keys):
        image_groups.setdefault(key, []).append(idx)
    content_residual = matrix.clone()
    for idxs in image_groups.values():
        idx_t = torch.as_tensor(idxs, dtype=torch.long, device=matrix.device)
        group_mean = matrix.index_select(0, idx_t).mean(dim=0, keepdim=True)
        content_residual.index_copy_(0, idx_t, matrix.index_select(0, idx_t) - group_mean + grand)
    target_after_source_image_ratio, _ = _between_ratio(target_style_ids, content_residual)

    target_norm_rows = {}
    for style_id in sorted(set(target_style_ids)):
        idxs = [idx for idx, label in enumerate(target_style_ids) if label == style_id]
        idx_t = torch.as_tensor(idxs, dtype=torch.long, device=matrix.device)
        target_mean = matrix.index_select(0, idx_t).mean(dim=0)
        target_norm_rows[classes[int(style_id)]] = {
            "count": len(idxs),
            "mean_delta_norm": float(target_mean.norm().item()),
            "within_delta_std": float(
                (matrix.index_select(0, idx_t) - target_mean.view(1, -1)).square().sum(dim=1).sqrt().mean().item()
            ),
        }

    return {
        "total_delta_ss": float(total_ss.item()),
        "target_between_ratio": target_ratio,
        "source_style_between_ratio": source_style_ratio,
        "source_image_between_ratio": source_image_ratio,
        "source_target_pair_between_ratio": pair_ratio,
        "target_after_source_image_ratio": target_after_source_image_ratio,
        "target_group_count": target_groups,
        "source_style_group_count": source_style_groups,
        "source_image_group_count": source_image_groups,
        "source_target_pair_group_count": pair_groups,
        "by_target_norm": target_norm_rows,
    }


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
    image_size: int,
    batch_size: int,
    device: torch.device,
    clip_gpu_tensor: bool = False,
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
            if clip_gpu_tensor:
                tensors = torch.stack([_load_image_tensor_01(p, image_size) for p in paths], dim=0)
                feats.append(encode_clip(tensors.to(device, non_blocking=True)).detach())
            else:
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


@torch.no_grad()
def _precompute_source_features(
    *,
    items: list[dict],
    encode_clip,
    image_size: int,
    batch_size: int,
    device: torch.device,
    timers: dict[str, float],
    profile_sync: bool,
    clip_gpu_tensor: bool = False,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    unique_paths = sorted({str(item["image_path"].resolve()) for item in items})
    clip_cache: dict[str, torch.Tensor] = {}
    lpips_cache: dict[str, torch.Tensor] = {}

    for start_idx in tqdm(range(0, len(unique_paths), batch_size), desc="clip/lpips source cache"):
        paths = [Path(p) for p in unique_paths[start_idx : start_idx + batch_size]]

        t0 = time.perf_counter()
        if clip_gpu_tensor:
            image_tensors = torch.stack([_load_image_tensor_01(path, image_size) for path in paths], dim=0)
            feats = encode_clip(image_tensors.to(device, non_blocking=True)).detach().cpu()
            _sync_if_needed(device, profile_sync)
        else:
            pils = []
            try:
                for path in paths:
                    with Image.open(path) as im:
                        pils.append(ImageOps.exif_transpose(im).convert("RGB"))
                feats = encode_clip(pils).detach().cpu()
                _sync_if_needed(device, profile_sync)
            finally:
                for pil in pils:
                    pil.close()
            image_tensors = torch.stack([_load_image_tensor_01(path, image_size) for path in paths], dim=0)
        _add_elapsed(timers, "source_clip", t0)

        t0 = time.perf_counter()
        for path, feat, image_tensor in zip(paths, feats, image_tensors):
            key = str(path.resolve())
            clip_cache[key] = feat
            lpips_cache[key] = image_tensor * 2.0 - 1.0
        _add_elapsed(timers, "source_lpips_load", t0)

    return clip_cache, lpips_cache


def _row_summary(rows: list[dict]) -> dict:
    if rows and "clip_style" not in rows[0]:
        cross_rows = [r for r in rows if r["src_style"] != r["tgt_style"]]
        identity_rows = [r for r in rows if r["src_style"] == r["tgt_style"]]
        return {
            "count": len(rows),
            "generate_only": True,
            "overall": {"count": len(rows)},
            "cross_only": {"count": len(cross_rows)},
            "identity_only": {"count": len(identity_rows)},
        }

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
    wall_start = time.perf_counter()
    timers: dict[str, float] = defaultdict(float)
    profile_sync = bool(args.profile_timing)
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

    generate_only = bool(args.generate_only)
    clip_gpu_tensor = bool(args.clip_gpu_tensor)
    encode_clip = None
    style_prototypes: dict[int, torch.Tensor] = {}
    lpips_model = None
    source_clip_cache: dict[str, torch.Tensor] = {}
    source_lpips_cache: dict[str, torch.Tensor] = {}

    if not generate_only:
        t0 = time.perf_counter()
        encode_clip = _load_clip_tensor(args, device) if clip_gpu_tensor else _load_clip(args, device)
        _sync_if_needed(device, profile_sync)
        _add_elapsed(timers, "load_clip", t0)

        t0 = time.perf_counter()
        style_prototypes = _compute_style_prototypes(
            image_root=image_root,
            classes=classes,
            encode_clip=encode_clip,
            image_size=int(args.image_size),
            batch_size=max(1, int(args.ref_batch_size)),
            device=device,
            clip_gpu_tensor=clip_gpu_tensor,
        )
        _sync_if_needed(device, profile_sync)
        _add_elapsed(timers, "style_prototypes", t0)

        import lpips

        t0 = time.perf_counter()
        lpips_model = lpips.LPIPS(net=args.lpips_net, verbose=False).to(device).eval()
        _sync_if_needed(device, profile_sync)
        _add_elapsed(timers, "load_lpips", t0)

    t0 = time.perf_counter()
    ort_vae = None
    if str(args.vae_onnx_decoder).strip():
        ort_vae = ORTVAEDecoder(
            args.vae_onnx_decoder,
            device_id=device.index or 0,
            use_tensorrt=bool(args.vae_onnx_tensorrt),
            trt_cache_dir=str(args.vae_onnx_trt_cache_dir),
        )
        vae = None
        timers["vae_onnx_providers"] = 0.0
        print("vae_onnx_providers=" + ",".join(ort_vae.providers))
    else:
        vae = load_vae(
            device=str(device),
            model_id=args.vae_model,
            cache_dir=str(args.hf_cache_dir),
            compile_decoder=bool(args.vae_compile_decoder),
            compile_method=str(args.vae_compile_method),
            compile_mode=str(args.vae_compile_mode),
            compile_fullgraph=bool(args.vae_compile_fullgraph),
            compile_cache_dir=str(args.vae_compile_cache_dir),
        )
    _sync_if_needed(device, profile_sync)
    _add_elapsed(timers, "load_vae", t0)

    t0 = time.perf_counter()
    infer = LGTInference(
        str(checkpoint),
        device=str(device),
        num_steps=int(args.num_steps),
        step_size=float(args.step_size),
        style_strength=float(args.style_strength),
    )
    _sync_if_needed(device, profile_sync)
    _add_elapsed(timers, "load_lancet", t0)

    if not generate_only:
        assert encode_clip is not None
        source_clip_cache, source_lpips_cache = _precompute_source_features(
            items=items,
            encode_clip=encode_clip,
            image_size=int(args.image_size),
            batch_size=max(1, int(args.source_feature_batch_size)),
            device=device,
            timers=timers,
            profile_sync=profile_sync,
            clip_gpu_tensor=clip_gpu_tensor,
        )
    rows: list[dict] = []
    metric_batch = max(1, int(args.batch_size))
    delta_sums: dict[int, torch.Tensor] = {}
    delta_counts: dict[int, int] = {}
    delta_vectors: list[torch.Tensor] = []
    delta_source_style_ids: list[int] = []
    delta_target_style_ids: list[int] = []
    delta_source_image_keys: list[str] = []

    target_chunk = max(1, min(len(classes), int(args.target_chunk_size)))
    default_decode_bs = max(1, metric_batch * target_chunk)
    vae_decode_bs = max(1, int(args.vae_decode_batch_size) if int(args.vae_decode_batch_size) > 0 else default_decode_bs)
    save_generated = bool(args.save_generated)
    save_workers = max(0, int(args.image_save_workers))
    save_pool = (
        concurrent.futures.ThreadPoolExecutor(max_workers=save_workers)
        if save_generated and save_workers > 0
        else None
    )
    save_futures: list[concurrent.futures.Future] = []

    try:
        with torch.inference_mode():
            for target_start in range(0, len(classes), target_chunk):
                target_ids_chunk = list(range(target_start, min(len(classes), target_start + target_chunk)))
                target_label = ",".join(classes[idx] for idx in target_ids_chunk)
                for start in tqdm(range(0, len(items), metric_batch), desc=f"generate -> {target_label}"):
                    batch = items[start : start + metric_batch]

                    t0 = time.perf_counter()
                    latents = torch.stack([_load_latent(item["latent_path"]) for item in batch], dim=0).to(device)
                    _add_elapsed(timers, "load_latents", t0)

                    t0 = time.perf_counter()
                    repeated_latents = latents.repeat(len(target_ids_chunk), 1, 1, 1)
                    target_ids = torch.cat(
                        [
                            torch.full((latents.shape[0],), target_id, dtype=torch.long, device=device)
                            for target_id in target_ids_chunk
                        ],
                        dim=0,
                    )
                    out_latents = infer.transfer_style(
                        repeated_latents,
                        target_style_id=target_ids,
                        num_steps=int(args.num_steps),
                    )
                    _sync_if_needed(device, profile_sync)
                    _add_elapsed(timers, "lancet_generate", t0)

                    if generate_only:
                        latent_delta_l2 = None
                        latent_delta_abs_mean = None
                        latent_delta_cpu = None
                    else:
                        t0 = time.perf_counter()
                        latent_delta = (out_latents.float() - repeated_latents.float()).flatten(1)
                        latent_delta_cpu = latent_delta.detach().cpu()
                        latent_delta_l2 = latent_delta.norm(dim=1).detach().cpu().numpy()
                        latent_delta_abs_mean = latent_delta.abs().mean(dim=1).detach().cpu().numpy()
                        _sync_if_needed(device, profile_sync)
                        _add_elapsed(timers, "latent_delta_metrics", t0)

                    t0 = time.perf_counter()
                    decoded_parts = []
                    for dec_start in range(0, out_latents.shape[0], vae_decode_bs):
                        dec_end = min(out_latents.shape[0], dec_start + vae_decode_bs)
                        part = out_latents[dec_start:dec_end]
                        if ort_vae is not None:
                            decoded_part = ort_vae.decode(part, scaling_factor=float(args.vae_scaling_factor))
                        else:
                            decoded_part = decode_latent(vae, part, device=str(device))
                        decoded_parts.append(decoded_part.detach())
                    decoded = torch.cat(decoded_parts, dim=0)
                    _sync_if_needed(device, profile_sync)
                    _add_elapsed(timers, "vae_decode", t0)

                    t0 = time.perf_counter()
                    decoded_cpu = decoded.detach().cpu() if (save_generated or (not generate_only and not clip_gpu_tensor)) else None
                    _add_elapsed(timers, "decoded_to_cpu", t0)

                    for local_target_idx, target_id in enumerate(target_ids_chunk):
                        target_name = classes[target_id]
                        offset = local_target_idx * len(batch)
                        decoded_slice = decoded[offset : offset + len(batch)]
                        decoded_cpu_slice = decoded_cpu[offset : offset + len(batch)] if decoded_cpu is not None else None
                        delta_l2_slice = latent_delta_l2[offset : offset + len(batch)] if latent_delta_l2 is not None else None
                        delta_abs_slice = (
                            latent_delta_abs_mean[offset : offset + len(batch)]
                            if latent_delta_abs_mean is not None
                            else None
                        )
                        if latent_delta_cpu is not None:
                            delta_cpu_slice = latent_delta_cpu[offset : offset + len(batch)]
                            delta_sum = delta_cpu_slice.sum(dim=0)
                            if target_id in delta_sums:
                                delta_sums[target_id] = delta_sums[target_id] + delta_sum
                            else:
                                delta_sums[target_id] = delta_sum
                            delta_counts[target_id] = delta_counts.get(target_id, 0) + int(delta_cpu_slice.shape[0])
                            for i, item in enumerate(batch):
                                delta_vectors.append(delta_cpu_slice[i].clone())
                                delta_source_style_ids.append(int(item["source_style_id"]))
                                delta_target_style_ids.append(int(target_id))
                                delta_source_image_keys.append(str(item["latent_path"]))

                        src_lpips = []
                        gen_names = []

                        t0 = time.perf_counter()
                        for i, item in enumerate(batch):
                            gen_name = f"{item['source_style']}__{item['stem']}__to__{target_name}.png"
                            gen_path = gen_dir / gen_name
                            gen_names.append(gen_name)
                            if save_generated:
                                decoded_cpu_i = decoded_cpu_slice[i] if decoded_cpu_slice is not None else decoded_slice[i].detach().cpu()
                                if save_pool is not None:
                                    save_futures.append(save_pool.submit(_save_png, decoded_cpu_i, gen_path))
                                else:
                                    _save_png(decoded_cpu_i, gen_path)
                            if not generate_only:
                                src_key = str(item["image_path"].resolve())
                                src_lpips.append(source_lpips_cache[src_key])
                        _add_elapsed(timers, "png_save_submit", t0)

                        if generate_only:
                            for i, item in enumerate(batch):
                                rows.append(
                                    {
                                        "src_style": item["source_style"],
                                        "tgt_style": target_name,
                                        "src_image": str(item["image_path"]),
                                        "src_latent": str(item["latent_path"]),
                                        "gen_image": str(gen_dir / gen_names[i]),
                                    }
                                )
                            continue

                        assert encode_clip is not None
                        assert lpips_model is not None
                        t0 = time.perf_counter()
                        if clip_gpu_tensor:
                            gen_clip = encode_clip(decoded_slice)
                        else:
                            gen_pils = [_pil_from_tensor_01(img) for img in decoded_cpu_slice]
                            try:
                                gen_clip = encode_clip(gen_pils)
                            finally:
                                for pil in gen_pils:
                                    pil.close()
                        src_clip = torch.stack(
                            [source_clip_cache[str(item["image_path"].resolve())] for item in batch],
                            dim=0,
                        ).to(device)
                        target_proto = style_prototypes[target_id].expand(gen_clip.shape[0], -1)
                        clip_style_tensor = F.cosine_similarity(gen_clip, target_proto, dim=-1)
                        clip_content_tensor = F.cosine_similarity(gen_clip, src_clip, dim=-1)
                        source_to_target_clip_tensor = F.cosine_similarity(src_clip, target_proto, dim=-1)
                        clip_style_gain_tensor = clip_style_tensor - source_to_target_clip_tensor
                        dir_gen = gen_clip - src_clip
                        dir_tgt = target_proto - src_clip
                        dir_gen = dir_gen / dir_gen.norm(dim=-1, keepdim=True).clamp_min(1e-8)
                        dir_tgt = dir_tgt / dir_tgt.norm(dim=-1, keepdim=True).clamp_min(1e-8)
                        clip_dir_tensor = F.cosine_similarity(dir_gen, dir_tgt, dim=-1)
                        clip_style = clip_style_tensor.detach().cpu().numpy()
                        clip_content = clip_content_tensor.detach().cpu().numpy()
                        source_to_target_clip = source_to_target_clip_tensor.detach().cpu().numpy()
                        clip_style_gain = clip_style_gain_tensor.detach().cpu().numpy()
                        clip_dir = clip_dir_tensor.detach().cpu().numpy()
                        _sync_if_needed(device, profile_sync)
                        _add_elapsed(timers, "clip_generated", t0)

                        t0 = time.perf_counter()
                        lp_gen = decoded_slice * 2.0 - 1.0
                        lp_src = torch.stack(src_lpips, dim=0).to(device)
                        content_lpips = lpips_model(lp_gen, lp_src).view(-1).detach().cpu().numpy()
                        _sync_if_needed(device, profile_sync)
                        _add_elapsed(timers, "lpips_generated", t0)

                        for i, item in enumerate(batch):
                            rows.append(
                                {
                                    "src_style": item["source_style"],
                                    "tgt_style": target_name,
                                    "src_image": str(item["image_path"]),
                                    "src_latent": str(item["latent_path"]),
                                    "gen_image": str(gen_dir / gen_names[i]),
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

    finally:
        if save_pool is not None:
            t0 = time.perf_counter()
            for future in concurrent.futures.as_completed(save_futures):
                future.result()
            save_pool.shutdown(wait=True)
            _add_elapsed(timers, "png_async_join", t0)

    csv_path = out_dir / "metrics.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        metric_fields = ["src_style", "tgt_style", "src_image", "src_latent", "gen_image"]
        if not generate_only:
            metric_fields += [
                "clip_style",
                "source_to_target_clip",
                "clip_style_gain",
                "clip_dir",
                "clip_content",
                "content_lpips",
                "latent_delta_l2",
                "latent_delta_abs_mean",
            ]
        writer = csv.DictWriter(f, fieldnames=metric_fields)
        writer.writeheader()
        writer.writerows(rows)

    summary = _row_summary(rows)
    delta_diag = _delta_diagnostics(delta_sums, delta_counts, classes) if not generate_only else {}
    delta_var = (
        _delta_variance_decomposition(
            delta_vectors,
            source_style_ids=delta_source_style_ids,
            target_style_ids=delta_target_style_ids,
            source_image_keys=delta_source_image_keys,
            classes=classes,
        )
        if not generate_only
        else {}
    )
    summary.update(
        {
            "checkpoint": str(checkpoint),
            "latent_root": str(latent_root),
            "image_root": str(image_root),
            "classes": classes,
            "num_steps": int(args.num_steps),
            "step_size": float(args.step_size),
            "style_strength": float(args.style_strength),
            "generate_only": bool(generate_only),
            "generated_delta_diagnostics": delta_diag,
            "generated_delta_variance_decomposition": delta_var,
            "timings_sec": {
                **{key: float(value) for key, value in sorted(timers.items())},
                "wall_total": float(time.perf_counter() - wall_start),
            },
        }
    )
    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    if save_generated and not generate_only:
        _make_grid(rows, image_root, classes, out_dir / "grid_first_per_class.png")
    print(json.dumps(summary["overall"], indent=2))
    print("timings_sec=" + json.dumps(summary["timings_sec"], sort_keys=True))
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
    parser.add_argument("--source-feature-batch-size", type=int, default=64)
    parser.add_argument("--max-per-source-style", type=int, default=0)
    parser.add_argument("--num-steps", type=int, default=4)
    parser.add_argument("--step-size", type=float, default=1.0)
    parser.add_argument("--style-strength", type=float, default=1.0)
    parser.add_argument("--vae-model", default="ema")
    parser.add_argument("--vae-scaling-factor", type=float, default=0.18215)
    parser.add_argument("--vae-onnx-decoder", default="")
    parser.add_argument("--vae-onnx-tensorrt", action="store_true")
    parser.add_argument("--vae-onnx-trt-cache-dir", default="")
    parser.add_argument("--vae-compile-decoder", action="store_true")
    parser.add_argument("--vae-compile-method", default="pt2", choices=["pt2", "jit"])
    parser.add_argument("--vae-compile-mode", default="reduce-overhead", choices=["default", "reduce-overhead", "max-autotune", "max-autotune-no-cudagraphs"])
    parser.add_argument("--vae-compile-fullgraph", action="store_true")
    parser.add_argument("--vae-compile-cache-dir", default="")
    parser.add_argument("--hf-cache-dir", default="I:/Github/Latent_Style/eval_cache/hf")
    parser.add_argument(
        "--clip-model",
        default="I:/Github/Latent_Style/eval_cache/manual_clip/openai-clip-vit-base-patch32",
    )
    parser.add_argument("--clip-allow-network", action="store_true")
    parser.add_argument("--clip-gpu-tensor", action="store_true")
    parser.add_argument("--lpips-net", default="vgg", choices=["vgg", "alex", "squeeze"])
    parser.add_argument("--profile-timing", action="store_true")
    parser.add_argument("--generate-only", action="store_true", help="Run LANCET + VAE decode only; skip CLIP/LPIPS metrics.")
    parser.add_argument("--save-generated", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--image-save-workers", type=int, default=4)
    return parser.parse_args()


def main() -> None:
    evaluate(parse_args())


if __name__ == "__main__":
    main()
