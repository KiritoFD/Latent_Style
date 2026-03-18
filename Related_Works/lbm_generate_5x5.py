import argparse
import json
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path

import torch
from PIL import Image
from torchvision.transforms import ToTensor


@dataclass(frozen=True)
class MemoryStats:
    max_allocated_mb: float
    max_reserved_mb: float


def _dtype_from_str(s: str) -> torch.dtype:
    s = str(s or "").strip().lower()
    if s in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if s in {"fp16", "float16"}:
        return torch.float16
    if s in {"fp32", "float32"}:
        return torch.float32
    raise ValueError(f"Invalid --dtype: {s}. Expected one of: bf16, fp16, fp32.")


def _list_dataset_images(root: Path):
    styles = sorted([d.name for d in root.iterdir() if d.is_dir()], key=lambda x: x.lower())
    src = []
    for s in styles:
        for p in sorted((root / s).iterdir()):
            if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}:
                src.append((s, p))
    return styles, src


def _parse_kv(items: list[str], *, flag: str) -> dict[str, str]:
    out: dict[str, str] = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"Invalid {flag}: {item}, expected key=value")
        k, v = item.split("=", 1)
        k = k.strip()
        v = v.strip()
        if not k or not v:
            raise ValueError(f"Invalid {flag}: {item}")
        out[k] = v
    return out


def _gpu_mem_stats() -> MemoryStats:
    if not torch.cuda.is_available():
        return MemoryStats(0.0, 0.0)
    alloc = float(torch.cuda.max_memory_allocated()) / (1024.0**2)
    resv = float(torch.cuda.max_memory_reserved()) / (1024.0**2)
    return MemoryStats(alloc, resv)


def _sanitize_cache_dir_name(s: str) -> str:
    s = str(s or "").strip()
    if not s:
        return "model"
    s = s.replace(":", "_").replace("\\", "_").replace("/", "__")
    s = re.sub(r"[^0-9A-Za-z._-]+", "_", s)
    return s[:120]


def _prepare_batch(
    model_source_key: str,
    src_pil: Image.Image,
    *,
    device: torch.device,
    dtype: torch.dtype,
    size: int,
    extra_images: dict[str, Image.Image],
    extra_masks: dict[str, Image.Image],
):
    def img_to_tensor_rgb(img: Image.Image) -> torch.Tensor:
        t = ToTensor()(img.convert("RGB").resize((size, size), Image.Resampling.LANCZOS)).unsqueeze(0)
        return t * 2 - 1

    def mask_to_tensor(img: Image.Image) -> torch.Tensor:
        m = img.convert("L").resize((size, size), Image.Resampling.BILINEAR)
        t = ToTensor()(m).unsqueeze(0)
        return t.clamp(0, 1)

    batch: dict[str, torch.Tensor] = {}
    batch[model_source_key] = img_to_tensor_rgb(src_pil).to(device=device, dtype=dtype, non_blocking=True)

    for k, img in extra_images.items():
        batch[k] = img_to_tensor_rgb(img).to(device=device, dtype=dtype, non_blocking=True)
    for k, img in extra_masks.items():
        batch[k] = mask_to_tensor(img).to(device=device, dtype=dtype, non_blocking=True)

    return batch


def _evaluate_one(
    model,
    src_pil: Image.Image,
    *,
    num_steps: int,
    device: torch.device,
    dtype: torch.dtype,
    size: int,
    out_size: int,
    extra_images: dict[str, Image.Image],
    extra_masks: dict[str, Image.Image],
):
    batch = _prepare_batch(
        model.source_key,
        src_pil,
        device=device,
        dtype=dtype,
        size=size,
        extra_images=extra_images,
        extra_masks=extra_masks,
    )
    z_source = model.vae.encode(batch[model.source_key])
    out = model.sample(
        z=z_source,
        num_steps=int(num_steps),
        conditioner_inputs=batch,
        max_samples=1,
    ).clamp(-1, 1)
    out = (out[0].float().cpu() + 1) / 2
    out_pil = Image.fromarray((out.permute(1, 2, 0).numpy() * 255.0).round().clip(0, 255).astype("uint8"))
    if int(out_size) > 0:
        out_pil = out_pil.resize((int(out_size), int(out_size)), Image.Resampling.LANCZOS)
    return out_pil


def _load_static_images(map_str: dict[str, str]) -> dict[str, Image.Image]:
    out: dict[str, Image.Image] = {}
    for k, v in map_str.items():
        p = Path(v)
        if not p.exists():
            raise FileNotFoundError(f"Missing file for {k}: {p}")
        out[k] = Image.open(p)
    return out


def main():
    ap = argparse.ArgumentParser("Generate 5x5 outputs using LBM (single-step image-to-image) with VRAM-safe execution")
    ap.add_argument("--test_dir", required=True, help="Dataset dir with style subfolders")
    ap.add_argument("--out_dir", required=True, help="Output dir (will create images/ and meta.json)")
    ap.add_argument(
        "--model_id",
        default="jasperai/LBM_depth",
        help="Default LBM model (HF repo id or local dir). Used for all target styles unless overridden by --model_for_style.",
    )
    ap.add_argument(
        "--model_for_style",
        action="append",
        default=[],
        help="Override per target style: e.g. --model_for_style monet=path_or_hf_id. Can repeat.",
    )
    ap.add_argument(
        "--ckpt_cache_dir",
        default=str(Path(__file__).resolve().parent / "external" / "LBM" / "examples" / "inference" / "ckpts"),
        help="Directory used to cache HF snapshots (per model).",
    )
    ap.add_argument("--num_steps", type=int, default=1, help="LBM sampling steps (NFE). Default 1.")
    ap.add_argument("--size", type=int, default=512, help="Internal inference resolution (square).")
    ap.add_argument("--out_size", type=int, default=256, help="Final saved resolution (square).")
    ap.add_argument("--size_fallbacks", default="384,256", help="Comma-separated sizes to try on CUDA OOM.")
    ap.add_argument("--dtype", default="bf16", help="bf16|fp16|fp32 (bf16 recommended if supported).")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--skip_identity", action="store_true")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--max_per_style", type=int, default=0, help="0 means all")
    ap.add_argument("--progress_every", type=int, default=10)
    ap.add_argument(
        "--extra_image",
        action="append",
        default=[],
        help="Extra conditioning image(s) passed to model as batch key: KEY=path.jpg (repeatable).",
    )
    ap.add_argument(
        "--extra_mask",
        action="append",
        default=[],
        help="Extra conditioning mask(s) passed to model as batch key: KEY=path.png (repeatable).",
    )
    args = ap.parse_args()

    test_dir = Path(args.test_dir).resolve()
    out_dir = Path(args.out_dir).resolve()
    img_dir = out_dir / "images"
    img_dir.mkdir(parents=True, exist_ok=True)

    styles, all_src = _list_dataset_images(test_dir)
    if args.max_per_style > 0:
        grouped: dict[str, list[tuple[str, Path]]] = {}
        for s, p in all_src:
            grouped.setdefault(s, []).append((s, p))
        all_src = []
        for s in styles:
            all_src.extend(grouped.get(s, [])[: int(args.max_per_style)])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = _dtype_from_str(args.dtype)
    if device.type != "cuda" and dtype != torch.float32:
        print(f"[lbm] device={device} -> forcing dtype=float32 (was {dtype})")
        dtype = torch.float32

    torch.manual_seed(int(args.seed))
    if device.type == "cuda":
        torch.cuda.manual_seed_all(int(args.seed))

    try_sizes = [int(args.size)]
    fallbacks = [s for s in str(args.size_fallbacks).split(",") if s.strip()]
    try_sizes.extend([int(s) for s in fallbacks])
    try_sizes = [s for s in try_sizes if s > 0]

    model_map = _parse_kv(args.model_for_style, flag="--model_for_style")
    per_style_model_id = {s: model_map.get(s, args.model_id) for s in styles}

    extra_images = _load_static_images(_parse_kv(args.extra_image, flag="--extra_image"))
    extra_masks = _load_static_images(_parse_kv(args.extra_mask, flag="--extra_mask"))

    # Lazy import so envs without LBM don't choke when just printing --help.
    from lbm.inference import get_model  # noqa: PLC0415

    print(f"[lbm] device={device} dtype={dtype} steps={int(args.num_steps)} size={try_sizes} out_size={int(args.out_size)}")
    print(f"[lbm] styles={styles} sources={len(all_src)} targets={len(styles)}")

    total = len(all_src) * len(styles)
    done = 0
    started = time.perf_counter()
    gen_count = 0
    infer_time_sec = 0.0
    oom_retries = 0
    per_target_peak_mb: dict[str, float] = {}

    styles_by_model_id: dict[str, list[str]] = {}
    for s in styles:
        styles_by_model_id.setdefault(per_style_model_id[s], []).append(s)

    for model_id, tgt_styles in styles_by_model_id.items():
        cache_dir = Path(args.ckpt_cache_dir).resolve() / _sanitize_cache_dir_name(model_id)
        cache_dir.mkdir(parents=True, exist_ok=True)

        print(f"[lbm] loading model: {model_id} (targets={tgt_styles})")
        model = get_model(
            model_id,
            save_dir=str(cache_dir),
            torch_dtype=dtype,
            device=str(device),
        )

        if device.type == "cuda":
            torch.cuda.empty_cache()

        for tgt_style in tgt_styles:
            peak_for_tgt = 0.0
            for src_style, src_path in all_src:
                out_name = f"{src_style}_{src_path.stem}_to_{tgt_style}.jpg"
                out_path = img_dir / out_name
                if out_path.exists() and not args.overwrite:
                    done += 1
                    continue

                src_img = Image.open(src_path).convert("RGB")
                if src_style == tgt_style and not args.skip_identity:
                    src_img.resize((int(args.out_size), int(args.out_size)), Image.Resampling.LANCZOS).save(
                        out_path,
                        quality=95,
                    )
                    done += 1
                    continue

                out_img = None
                last_err = None
                for size in try_sizes:
                    try:
                        if device.type == "cuda":
                            torch.cuda.reset_peak_memory_stats()
                        t0 = time.perf_counter()
                        with torch.inference_mode():
                            out_img = _evaluate_one(
                                model,
                                src_img,
                                num_steps=int(args.num_steps),
                                device=device,
                                dtype=dtype,
                                size=int(size),
                                out_size=int(args.out_size),
                                extra_images=extra_images,
                                extra_masks=extra_masks,
                            )
                        infer_time_sec += time.perf_counter() - t0
                        gen_count += 1
                        if device.type == "cuda":
                            m = _gpu_mem_stats()
                            peak_for_tgt = max(peak_for_tgt, m.max_allocated_mb)
                        break
                    except RuntimeError as e:
                        last_err = e
                        if "out of memory" not in str(e).lower() or device.type != "cuda":
                            raise
                        oom_retries += 1
                        torch.cuda.empty_cache()
                        continue
                    finally:
                        if device.type == "cuda":
                            torch.cuda.empty_cache()

                if out_img is None:
                    raise RuntimeError(f"LBM failed for {src_path} -> {tgt_style}. Last error: {last_err}")

                out_img.save(out_path, quality=95)
                done += 1
                if args.progress_every > 0 and done % int(args.progress_every) == 0:
                    mins = (time.perf_counter() - started) / 60.0
                    print(f"[lbm] {done}/{total} ({mins:.1f} min) tgt={tgt_style}")

            per_target_peak_mb[tgt_style] = float(peak_for_tgt)

        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    elapsed = time.perf_counter() - started
    meta = {
        "method": "lbm",
        "device": str(device),
        "dtype": str(dtype),
        "model_id_default": args.model_id,
        "model_id_per_target_style": per_style_model_id,
        "ckpt_cache_dir": str(Path(args.ckpt_cache_dir).resolve()),
        "num_steps": int(args.num_steps),
        "size": int(args.size),
        "size_fallbacks": [int(s) for s in try_sizes[1:]],
        "out_size": int(args.out_size),
        "seed": int(args.seed),
        "skip_identity": bool(args.skip_identity),
        "styles": styles,
        "source_count": len(all_src),
        "generated_images": int(gen_count),
        "oom_retries": int(oom_retries),
        "elapsed_sec": float(elapsed),
        "inference_sec": float(infer_time_sec),
        "avg_infer_sec_per_generated": float(infer_time_sec / max(gen_count, 1)),
        "cuda_peak_allocated_mb_per_target": per_target_peak_mb,
        "extra_images": {k: str(v.filename) if hasattr(v, "filename") else "" for k, v in extra_images.items()},
        "extra_masks": {k: str(v.filename) if hasattr(v, "filename") else "" for k, v in extra_masks.items()},
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    (out_dir / "summary.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[lbm] done -> {img_dir}")


if __name__ == "__main__":
    main()
