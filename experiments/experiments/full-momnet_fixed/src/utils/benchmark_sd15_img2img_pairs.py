from __future__ import annotations

import argparse
import csv
import json
import random
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from diffusers import StableDiffusionImg2ImgPipeline

IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


STYLE_PROMPTS = {
    "photo": "high quality realistic photo, natural lighting, detailed",
    "Hayao": "in the style of Hayao Miyazaki, studio ghibli anime background art",
    "monet": "in the style of Claude Monet, impressionist oil painting",
    "cezanne": "in the style of Paul Cezanne, post-impressionist painting",
    "vangogh": "in the style of Vincent van Gogh, expressive brushstrokes, oil painting",
}


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _resolve_path(raw: str, config_path: Path) -> Path:
    p = Path(raw).expanduser()
    if p.is_absolute():
        return p
    bases = [
        config_path.parent.resolve(),
        config_path.parent.parent.resolve(),
        Path.cwd().resolve(),
        Path(__file__).resolve().parents[2],
    ]
    for b in bases:
        cand = (b / p).resolve()
        if cand.exists():
            return cand
    return (bases[0] / p).resolve()


def _list_images(d: Path) -> list[Path]:
    if not d.exists():
        return []
    return sorted([p for p in d.iterdir() if p.is_file() and p.suffix.lower() in IMG_EXTS])


def _load_image(path: Path, size: int) -> Image.Image:
    return Image.open(path).convert("RGB").resize((size, size))


def _pick_style_names(cfg: dict, test_dir: Path) -> list[str]:
    names = list(cfg.get("data", {}).get("style_subdirs", []))
    # keep only the requested 5 styles if present
    preferred = ["photo", "Hayao", "monet", "vangogh", "cezanne"]
    if names:
        names = [s for s in preferred if s in names]
        if names:
            return names
    return [d.name for d in sorted(test_dir.iterdir()) if d.is_dir() and d.name in preferred]


def main() -> None:
    ap = argparse.ArgumentParser("SD1.5 img2img pairwise performance benchmark")
    ap.add_argument("--config", type=str, default=str(Path(__file__).resolve().parents[1] / "config.json"))
    ap.add_argument("--test_dir", type=str, default="", help="Override config.training.test_image_dir")
    ap.add_argument("--sd15_model_id", type=str, default="runwayml/stable-diffusion-v1-5")
    ap.add_argument("--image_size", type=int, default=256)
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--strength", type=float, default=0.75)
    ap.add_argument("--guidance_scale", type=float, default=7.5)
    ap.add_argument("--batch_size", type=int, default=1)
    ap.add_argument("--max_src_samples", type=int, default=50)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--save_outputs", action="store_true")
    ap.add_argument("--run_full_eval", action="store_true", help="Run run_evaluation.py with --reuse_generated after SD1.5 generation")
    ap.add_argument("--eval_checkpoint", type=str, default="", help="Checkpoint for run_evaluation full_eval context/config")
    ap.add_argument("--eval_batch_size", type=int, default=8)
    ap.add_argument("--eval_enable_art_fid", action="store_true")
    ap.add_argument("--eval_image_classifier_path", type=str, default="")
    ap.add_argument("--out_dir", type=str, default="sd15_img2img_pairs")
    ap.add_argument("--out_json", type=str, default="")
    ap.add_argument("--out_csv", type=str, default="")
    args = ap.parse_args()

    random.seed(int(args.seed))
    np.random.seed(int(args.seed))
    torch.manual_seed(int(args.seed))

    config_path = Path(args.config).resolve()
    cfg = json.loads(config_path.read_text(encoding="utf-8"))
    test_dir_raw = str(args.test_dir).strip() or str(cfg.get("training", {}).get("test_image_dir", "")).strip()
    if not test_dir_raw:
        raise ValueError("test_dir is empty and config.training.test_image_dir is empty")
    test_dir = _resolve_path(test_dir_raw, config_path)
    if not test_dir.exists():
        raise FileNotFoundError(f"test_dir not found: {test_dir}")

    style_names = _pick_style_names(cfg, test_dir)
    if len(style_names) < 2:
        raise RuntimeError(f"Need at least two style dirs in {test_dir}, got {style_names}")

    src_map: dict[str, list[Path]] = {}
    for s in style_names:
        ims = _list_images(test_dir / s)
        if int(args.max_src_samples) > 0:
            ims = ims[: int(args.max_src_samples)]
        src_map[s] = ims

    total_src = sum(len(v) for v in src_map.values())
    if total_src == 0:
        raise RuntimeError(f"No images found under test_dir: {test_dir}")

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available")

    dtype = torch.float16 if device.type == "cuda" else torch.float32
    pipe = StableDiffusionImg2ImgPipeline.from_pretrained(str(args.sd15_model_id), torch_dtype=dtype).to(device)
    pipe.safety_checker = None
    pipe.set_progress_bar_config(disable=True)
    pipe.enable_attention_slicing()

    # warmup once
    warm_s = next(iter(src_map.keys()))
    if not src_map[warm_s]:
        raise RuntimeError("Warmup source style has no images")
    warm_img = _load_image(src_map[warm_s][0], int(args.image_size))
    warm_prompt = STYLE_PROMPTS.get(style_names[0], f"in the style of {style_names[0]}")
    neg = "low quality, blurry, distorted, artifacts"
    gen = torch.Generator(device=str(device)).manual_seed(int(args.seed))
    _ = pipe(
        prompt=warm_prompt,
        image=warm_img,
        num_inference_steps=int(args.steps),
        strength=float(args.strength),
        guidance_scale=float(args.guidance_scale),
        negative_prompt=neg,
        generator=gen,
    ).images[0]
    _sync(device)

    out_dir = Path(args.out_dir).resolve()
    if args.save_outputs:
        out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    global_count = 0
    global_time = 0.0
    global_peak = 0.0

    bs = max(1, int(args.batch_size))
    print(f"[sd15-pairs] test_dir={test_dir}")
    print(f"[sd15-pairs] styles={style_names}")
    print(f"[sd15-pairs] steps={args.steps} strength={args.strength} guidance={args.guidance_scale} batch={bs}")

    for src_style in style_names:
        src_images = src_map[src_style]
        if not src_images:
            continue
        for tgt_style in style_names:
            prompt = STYLE_PROMPTS.get(tgt_style, f"in the style of {tgt_style}")
            if device.type == "cuda":
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats(device)

            t0 = time.perf_counter()
            pair_count = 0
            for i in range(0, len(src_images), bs):
                part = src_images[i : i + bs]
                ims = [_load_image(p, int(args.image_size)) for p in part]
                prompts = [prompt] * len(part)
                outs = pipe(
                    prompt=prompts,
                    image=ims,
                    num_inference_steps=int(args.steps),
                    strength=float(args.strength),
                    guidance_scale=float(args.guidance_scale),
                    negative_prompt=[neg] * len(part),
                    generator=gen,
                ).images
                _sync(device)
                if args.save_outputs:
                    for j, im in enumerate(outs):
                        src_stem = part[j].stem
                        im.save(out_dir / f"{src_style}_{src_stem}_to_{tgt_style}.jpg")
                pair_count += len(part)

            dt = time.perf_counter() - t0
            peak = float(torch.cuda.max_memory_allocated(device) / (1024**3)) if device.type == "cuda" else 0.0
            global_peak = max(global_peak, peak)
            global_count += pair_count
            global_time += dt
            ms = 1000.0 * dt / max(1, pair_count)
            ips = pair_count / max(1e-8, dt)
            row = {
                "src_style": src_style,
                "tgt_style": tgt_style,
                "count": int(pair_count),
                "total_sec": float(dt),
                "avg_ms_per_img": float(ms),
                "ips": float(ips),
                "peak_alloc_gb": float(peak),
            }
            rows.append(row)
            print(
                f"[sd15-pairs] {src_style:>8s}->{tgt_style:<8s} "
                f"count={pair_count:4d} ms/img={ms:8.2f} ips={ips:6.2f} peak_gb={peak:5.3f}"
            )

    overall = {
        "total_images": int(global_count),
        "total_sec": float(global_time),
        "avg_ms_per_img": float(1000.0 * global_time / max(1, global_count)),
        "ips": float(global_count / max(1e-8, global_time)),
        "peak_alloc_gb_max": float(global_peak),
    }

    print("[sd15-pairs] -------- overall --------")
    print(
        f"[sd15-pairs] total={overall['total_images']} "
        f"avg_ms/img={overall['avg_ms_per_img']:.2f} ips={overall['ips']:.2f} "
        f"peak_gb(max)={overall['peak_alloc_gb_max']:.3f}"
    )

    result = {
        "config": {
            "test_dir": str(test_dir),
            "styles": style_names,
            "sd15_model_id": str(args.sd15_model_id),
            "image_size": int(args.image_size),
            "steps": int(args.steps),
            "strength": float(args.strength),
            "guidance_scale": float(args.guidance_scale),
            "batch_size": int(bs),
            "max_src_samples": int(args.max_src_samples),
            "save_outputs": bool(args.save_outputs),
            "out_dir": str(out_dir),
        },
        "pairs": rows,
        "overall": overall,
    }

    out_json = Path(args.out_json).resolve() if str(args.out_json).strip() else out_dir / "benchmark_sd15_pairs.json"
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"[sd15-pairs] json: {out_json}")

    out_csv = Path(args.out_csv).resolve() if str(args.out_csv).strip() else out_dir / "benchmark_sd15_pairs.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["src_style", "tgt_style", "count", "total_sec", "avg_ms_per_img", "ips", "peak_alloc_gb"],
        )
        writer.writeheader()
        for r in rows:
            writer.writerow(r)
    print(f"[sd15-pairs] csv: {out_csv}")

    if args.run_full_eval:
        eval_ckpt = str(args.eval_checkpoint).strip()
        if not eval_ckpt:
            raise ValueError("--run_full_eval requires --eval_checkpoint")
        eval_ckpt_path = Path(eval_ckpt).resolve()
        if not eval_ckpt_path.exists():
            raise FileNotFoundError(f"eval checkpoint not found: {eval_ckpt_path}")

        run_eval = Path(__file__).resolve().parent / "run_evaluation.py"
        cmd = [
            sys.executable,
            str(run_eval),
            "--checkpoint",
            str(eval_ckpt_path),
            "--output",
            str(out_dir),
            "--test_dir",
            str(test_dir),
            "--batch_size",
            str(int(args.eval_batch_size)),
            "--reuse_generated",
            "--force_regen",
            "--classifier_path",
            "",
        ]
        if args.eval_enable_art_fid:
            cmd += ["--eval_enable_art_fid"]
        image_cls = str(args.eval_image_classifier_path).strip() or str(
            cfg.get("training", {}).get("full_eval_image_classifier_path", "")
        ).strip()
        if image_cls:
            cmd += ["--image_classifier_path", image_cls]

        print("[sd15-pairs] running full_eval on generated SD1.5 outputs ...")
        subprocess.run(cmd, check=True, cwd=str(Path(__file__).resolve().parent.parent))

        summary_path = out_dir / "summary.json"
        if summary_path.exists():
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            ana = summary.get("analysis", {})
            sty = ana.get("style_transfer_ability", {})
            p2a = ana.get("photo_to_art_performance", {})
            print("[sd15-pairs] -------- full_eval summary --------")
            print(
                "[sd15-pairs] style_transfer_ability | "
                f"classifier_acc={sty.get('classifier_acc')} "
                f"content_lpips={sty.get('content_lpips')} "
                f"clip_dir={sty.get('clip_dir')} "
                f"art_fid={sty.get('art_fid')}"
            )
            print(
                "[sd15-pairs] photo_to_art_performance | "
                f"classifier_acc={p2a.get('classifier_acc')} "
                f"clip_dir={p2a.get('clip_dir')} "
                f"art_fid={p2a.get('art_fid')}"
            )
            result["full_eval_summary"] = {
                "style_transfer_ability": sty,
                "photo_to_art_performance": p2a,
                "summary_path": str(summary_path),
            }
            out_json.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
            print(f"[sd15-pairs] updated json with full_eval summary: {out_json}")
        else:
            print(f"[sd15-pairs] WARNING: summary.json not found: {summary_path}")


if __name__ == "__main__":
    main()
