""""

[cmp] -------- result --------
[cmp] AdaCUT avg_total_ms/img: 43.04 | ips=23.23
[cmp]   breakdown enc/gen/dec: 11.73 / 5.91 / 25.41 ms
[cmp] SD1.5 avg_total_ms/img: 728.75 | ips=1.37
[cmp] speedup (SD15/AdaCUT): 16.93x
[cmp] AdaCUT peak_alloc_gb: 0.355
[cmp] SD1.5 peak_alloc_gb: 2.170
[cmp] vram ratio (SD15/AdaCUT): 6.12x
"""

from __future__ import annotations

import argparse
import json
import random
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from diffusers import StableDiffusionImg2ImgPipeline
from utils.inference import LGTInference, decode_latent, encode_image, load_vae

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


def _collect_images(root: Path) -> list[Path]:
    if not root.exists():
        return []
    if root.is_file() and root.suffix.lower() in IMG_EXTS:
        return [root]
    return sorted([p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in IMG_EXTS])


def _load_tensor_256(path: Path) -> torch.Tensor:
    img = Image.open(path).convert("RGB").resize((256, 256))
    arr = np.asarray(img, dtype=np.float32) / 255.0
    x = torch.from_numpy(arr).permute(2, 0, 1).contiguous()
    return x * 2.0 - 1.0


def _load_pil(path: Path, size: int) -> Image.Image:
    return Image.open(path).convert("RGB").resize((size, size))


def benchmark_lgt(
    checkpoint: Path,
    image_paths: list[Path],
    target_style_id: int,
    device: torch.device,
    *,
    num_steps: int,
    step_size: float,
    style_strength: float,
) -> dict:
    inf = LGTInference(
        str(checkpoint),
        device=str(device),
        num_steps=int(num_steps),
        step_size=float(step_size),
        style_strength=float(style_strength),
    )
    vae = load_vae(device=str(device))
    model_dtype = next(inf.model.parameters()).dtype
    model_scale = float(getattr(inf.model, "latent_scale_factor", 0.18215))
    vae_scale = float(getattr(getattr(vae, "config", None), "scaling_factor", model_scale))
    scale_in = model_scale / max(vae_scale, 1e-8)
    scale_out = vae_scale / max(model_scale, 1e-8)

    # warmup
    xb = _load_tensor_256(image_paths[0]).unsqueeze(0).to(device)
    _sync(device)
    z = encode_image(vae, xb, device=str(device))
    if abs(scale_in - 1.0) > 1e-4:
        z = z * scale_in
    z = z.to(device=device, dtype=model_dtype)
    tgt = torch.full((1,), int(target_style_id), dtype=torch.long, device=device)
    zt = inf.generation(z, target_style_id=tgt, num_steps=int(num_steps))
    if abs(scale_out - 1.0) > 1e-4:
        zt = zt * scale_out
    _ = decode_latent(vae, zt, device=str(device))
    _sync(device)

    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)

    t_total = 0.0
    t_enc = 0.0
    t_gen = 0.0
    t_dec = 0.0
    for p in image_paths:
        xb = _load_tensor_256(p).unsqueeze(0).to(device)
        _sync(device)
        t0 = time.perf_counter()
        t1 = time.perf_counter()
        z = encode_image(vae, xb, device=str(device))
        if abs(scale_in - 1.0) > 1e-4:
            z = z * scale_in
        z = z.to(device=device, dtype=model_dtype)
        _sync(device)
        t2 = time.perf_counter()
        tgt = torch.full((1,), int(target_style_id), dtype=torch.long, device=device)
        zt = inf.generation(z, target_style_id=tgt, num_steps=int(num_steps))
        if abs(scale_out - 1.0) > 1e-4:
            zt = zt * scale_out
        _sync(device)
        t3 = time.perf_counter()
        _ = decode_latent(vae, zt, device=str(device))
        _sync(device)
        t4 = time.perf_counter()
        t_total += t4 - t0
        t_enc += t2 - t1
        t_gen += t3 - t2
        t_dec += t4 - t3

    n = len(image_paths)
    peak_gb = float(torch.cuda.max_memory_allocated(device) / (1024**3)) if device.type == "cuda" else 0.0
    return {
        "name": "AdaCUT",
        "num_images": n,
        "avg_total_ms": 1000.0 * t_total / n,
        "avg_encode_ms": 1000.0 * t_enc / n,
        "avg_generation_ms": 1000.0 * t_gen / n,
        "avg_decode_ms": 1000.0 * t_dec / n,
        "throughput_ips": n / max(t_total, 1e-8),
        "peak_alloc_gb": peak_gb,
    }


def benchmark_sd15(
    image_paths: list[Path],
    prompt: str,
    device: torch.device,
    *,
    model_id: str,
    steps: int,
    strength: float,
    guidance_scale: float,
    image_size: int,
) -> dict:
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    pipe = StableDiffusionImg2ImgPipeline.from_pretrained(model_id, torch_dtype=dtype)
    pipe = pipe.to(device)
    pipe.safety_checker = None
    pipe.set_progress_bar_config(disable=True)
    pipe.enable_attention_slicing()

    neg = "low quality, blurry, distorted, artifacts"
    gen = torch.Generator(device=str(device)).manual_seed(42)

    # warmup
    img = _load_pil(image_paths[0], image_size)
    _ = pipe(
        prompt=prompt,
        image=img,
        num_inference_steps=int(steps),
        strength=float(strength),
        guidance_scale=float(guidance_scale),
        negative_prompt=neg,
        generator=gen,
    ).images[0]
    _sync(device)

    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)

    t_total = 0.0
    for p in image_paths:
        img = _load_pil(p, image_size)
        _sync(device)
        t0 = time.perf_counter()
        _ = pipe(
            prompt=prompt,
            image=img,
            num_inference_steps=int(steps),
            strength=float(strength),
            guidance_scale=float(guidance_scale),
            negative_prompt=neg,
            generator=gen,
        ).images[0]
        _sync(device)
        t1 = time.perf_counter()
        t_total += t1 - t0

    n = len(image_paths)
    peak_gb = float(torch.cuda.max_memory_allocated(device) / (1024**3)) if device.type == "cuda" else 0.0
    return {
        "name": "SD1.5_img2img",
        "num_images": n,
        "avg_total_ms": 1000.0 * t_total / n,
        "throughput_ips": n / max(t_total, 1e-8),
        "peak_alloc_gb": peak_gb,
    }


def _load_eval_sources(test_dir: Path, style_subdirs: list[str], max_src_samples: int) -> list[dict]:
    src = []
    rng = random.Random(42)
    for s in style_subdirs:
        sdir = test_dir / s
        if not sdir.exists():
            continue
        imgs = sorted([p for p in sdir.iterdir() if p.is_file() and p.suffix.lower() in IMG_EXTS])
        rng.shuffle(imgs)
        if max_src_samples > 0:
            imgs = imgs[:max_src_samples]
        for p in imgs:
            src.append({"style_name": s, "path": p})
    return src


def generate_sd15_full_eval_images(
    out_dir: Path,
    test_dir: Path,
    style_subdirs: list[str],
    *,
    model_id: str,
    steps: int,
    strength: float,
    guidance_scale: float,
    size: int,
    batch_size: int,
    max_src_samples: int,
    device: torch.device,
) -> int:
    out_dir.mkdir(parents=True, exist_ok=True)
    src_items = _load_eval_sources(test_dir, style_subdirs, max_src_samples=max_src_samples)
    if not src_items:
        raise RuntimeError(f"No source images found under {test_dir}")

    dtype = torch.float16 if device.type == "cuda" else torch.float32
    pipe = StableDiffusionImg2ImgPipeline.from_pretrained(model_id, torch_dtype=dtype).to(device)
    pipe.safety_checker = None
    pipe.set_progress_bar_config(disable=True)
    pipe.enable_attention_slicing()

    neg = "low quality, blurry, distorted, artifacts"
    gen = torch.Generator(device=str(device)).manual_seed(42)
    total = 0

    bs = max(1, int(batch_size))
    for tgt in style_subdirs:
        prompt = STYLE_PROMPTS.get(tgt, f"in the style of {tgt}")
        for i in range(0, len(src_items), bs):
            part = src_items[i : i + bs]
            ims = [_load_pil(x["path"], size) for x in part]
            prompts = [prompt] * len(part)
            out = pipe(
                prompt=prompts,
                image=ims,
                num_inference_steps=int(steps),
                strength=float(strength),
                guidance_scale=float(guidance_scale),
                negative_prompt=[neg] * len(part),
                generator=gen,
            ).images
            for j, im in enumerate(out):
                src_style = part[j]["style_name"]
                src_stem = Path(part[j]["path"]).stem
                out_name = f"{src_style}_{src_stem}_to_{tgt}.jpg"
                im.save(out_dir / out_name)
                total += 1
    return total


def run_full_eval(
    checkpoint: Path,
    output_dir: Path,
    test_dir: Path,
    *,
    reuse_generated: bool,
    force_regen: bool,
    batch_size: int,
    image_classifier_path: str | None = None,
    eval_enable_art_fid: bool = True,
) -> None:
    run_eval = Path(__file__).resolve().parent / "run_evaluation.py"
    cmd = [
        sys.executable,
        str(run_eval),
        "--checkpoint",
        str(checkpoint),
        "--output",
        str(output_dir),
        "--test_dir",
        str(test_dir),
        "--batch_size",
        str(int(batch_size)),
        "--classifier_path",
        "",
    ]
    if reuse_generated:
        cmd += ["--reuse_generated"]
    if force_regen:
        cmd += ["--force_regen"]
    if eval_enable_art_fid:
        cmd += ["--eval_enable_art_fid"]
    if image_classifier_path:
        cmd += ["--image_classifier_path", str(image_classifier_path)]
    subprocess.run(cmd, check=True, cwd=str(Path(__file__).resolve().parent.parent))


def _read_summary(summary_path: Path) -> dict:
    if not summary_path.exists():
        return {}
    return json.loads(summary_path.read_text(encoding="utf-8"))


def _metric(summary: dict, *keys, default=None):
    cur = summary
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


def main() -> None:
    ap = argparse.ArgumentParser("Compare inference speed: AdaCUT vs SD1.5 img2img")
    ap.add_argument("--config", type=str, default=str(Path(__file__).resolve().parents[1] / "config.json"))
    ap.add_argument("--checkpoint", type=str, required=True)
    ap.add_argument("--num_images", type=int, default=20)
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--target_style", type=str, default="Hayao")
    ap.add_argument("--target_style_id", type=int, default=-1)
    ap.add_argument("--sd15_model_id", type=str, default="runwayml/stable-diffusion-v1-5")
    ap.add_argument("--sd15_steps", type=int, default=20)
    ap.add_argument("--sd15_strength", type=float, default=0.75)
    ap.add_argument("--sd15_guidance_scale", type=float, default=7.5)
    ap.add_argument("--sd15_size", type=int, default=256)
    ap.add_argument("--full_eval_compare", action="store_true", help="Run full_eval comparison (AdaCUT vs SD1.5)")
    ap.add_argument("--full_eval_out_root", type=str, default="full_eval_compare")
    ap.add_argument("--full_eval_batch_size", type=int, default=8)
    ap.add_argument("--sd15_gen_batch_size", type=int, default=4)
    ap.add_argument("--full_eval_max_src_samples", type=int, default=50)
    ap.add_argument("--out_json", type=str, default="")
    args = ap.parse_args()

    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    config_path = Path(args.config).resolve()
    cfg = json.loads(config_path.read_text(encoding="utf-8"))
    test_dir_raw = str(cfg.get("training", {}).get("test_image_dir", "")).strip()
    test_dir = _resolve_path(test_dir_raw, config_path)
    imgs = _collect_images(test_dir)
    if not imgs:
        raise FileNotFoundError(f"No test images under {test_dir}")
    n = min(max(1, int(args.num_images)), len(imgs))
    imgs = imgs[:n]

    style_names = list(cfg.get("data", {}).get("style_subdirs", []))
    if int(args.target_style_id) >= 0:
        style_id = int(args.target_style_id)
        style_name = style_names[style_id] if 0 <= style_id < len(style_names) else str(style_id)
    else:
        style_name = str(args.target_style)
        style_id = style_names.index(style_name) if style_name in style_names else 0
    prompt = STYLE_PROMPTS.get(style_name, f"in the style of {style_name}")

    device = torch.device(args.device)
    print(f"[cmp] checkpoint={Path(args.checkpoint).resolve()}")
    print(f"[cmp] test_dir={test_dir} num_images={n}")
    print(f"[cmp] target_style={style_name} id={style_id}")
    print(f"[cmp] sd15_prompt={prompt}")

    lgt = benchmark_lgt(
        checkpoint=Path(args.checkpoint).resolve(),
        image_paths=imgs,
        target_style_id=style_id,
        device=device,
        num_steps=int(cfg.get("inference", {}).get("num_steps", 1)),
        step_size=float(cfg.get("inference", {}).get("step_size", 1.0)),
        style_strength=float(cfg.get("inference", {}).get("style_strength", 0.75)),
    )
    sd = benchmark_sd15(
        image_paths=imgs,
        prompt=prompt,
        device=device,
        model_id=str(args.sd15_model_id),
        steps=int(args.sd15_steps),
        strength=float(args.sd15_strength),
        guidance_scale=float(args.sd15_guidance_scale),
        image_size=int(args.sd15_size),
    )

    speedup = sd["avg_total_ms"] / max(lgt["avg_total_ms"], 1e-8)
    vram_ratio = sd["peak_alloc_gb"] / max(lgt["peak_alloc_gb"], 1e-8) if device.type == "cuda" else 0.0
    result = {
        "lgt": lgt,
        "sd15": sd,
        "speedup_lgt_vs_sd15": speedup,
        "vram_ratio_sd15_over_lgt": vram_ratio,
    }

    print("[cmp] -------- result --------")
    print(f"[cmp] AdaCUT avg_total_ms/img: {lgt['avg_total_ms']:.2f} | ips={lgt['throughput_ips']:.2f}")
    print(
        f"[cmp]   breakdown enc/gen/dec: {lgt['avg_encode_ms']:.2f} / "
        f"{lgt['avg_generation_ms']:.2f} / {lgt['avg_decode_ms']:.2f} ms"
    )
    print(f"[cmp] SD1.5 avg_total_ms/img: {sd['avg_total_ms']:.2f} | ips={sd['throughput_ips']:.2f}")
    print(f"[cmp] speedup (SD15/AdaCUT): {speedup:.2f}x")
    if device.type == "cuda":
        print(f"[cmp] AdaCUT peak_alloc_gb: {lgt['peak_alloc_gb']:.3f}")
        print(f"[cmp] SD1.5 peak_alloc_gb: {sd['peak_alloc_gb']:.3f}")
        print(f"[cmp] vram ratio (SD15/AdaCUT): {vram_ratio:.2f}x")

    if args.full_eval_compare:
        out_root = Path(args.full_eval_out_root).resolve()
        out_lgt = out_root / "adacut"
        out_sd = out_root / "sd15"
        out_root.mkdir(parents=True, exist_ok=True)
        image_classifier_path = str(cfg.get("training", {}).get("full_eval_image_classifier_path", "")).strip() or None

        print("[cmp][full_eval] running AdaCUT full_eval ...")
        run_full_eval(
            checkpoint=Path(args.checkpoint).resolve(),
            output_dir=out_lgt,
            test_dir=test_dir,
            reuse_generated=False,
            force_regen=True,
            batch_size=int(args.full_eval_batch_size),
            image_classifier_path=image_classifier_path,
            eval_enable_art_fid=True,
        )

        print("[cmp][full_eval] generating SD1.5 images on overfit50 ...")
        n_gen = generate_sd15_full_eval_images(
            out_dir=out_sd,
            test_dir=test_dir,
            style_subdirs=style_names,
            model_id=str(args.sd15_model_id),
            steps=int(args.sd15_steps),
            strength=float(args.sd15_strength),
            guidance_scale=float(args.sd15_guidance_scale),
            size=int(args.sd15_size),
            batch_size=int(args.sd15_gen_batch_size),
            max_src_samples=int(args.full_eval_max_src_samples),
            device=device,
        )
        print(f"[cmp][full_eval] SD1.5 generated images: {n_gen}")

        print("[cmp][full_eval] running SD1.5 full_eval with reuse_generated ...")
        run_full_eval(
            checkpoint=Path(args.checkpoint).resolve(),
            output_dir=out_sd,
            test_dir=test_dir,
            reuse_generated=True,
            force_regen=True,
            batch_size=int(args.full_eval_batch_size),
            image_classifier_path=image_classifier_path,
            eval_enable_art_fid=True,
        )

        sum_lgt = _read_summary(out_lgt / "summary.json")
        sum_sd = _read_summary(out_sd / "summary.json")
        lgt_acc = _metric(sum_lgt, "analysis", "style_transfer_ability", "classifier_acc", default=None)
        sd_acc = _metric(sum_sd, "analysis", "style_transfer_ability", "classifier_acc", default=None)
        lgt_lp = _metric(sum_lgt, "analysis", "style_transfer_ability", "content_lpips", default=None)
        sd_lp = _metric(sum_sd, "analysis", "style_transfer_ability", "content_lpips", default=None)
        lgt_artfid = _metric(sum_lgt, "analysis", "style_transfer_ability", "art_fid", default=None)
        sd_artfid = _metric(sum_sd, "analysis", "style_transfer_ability", "art_fid", default=None)
        print("[cmp][full_eval] -------- summary --------")
        print(f"[cmp][full_eval] AdaCUT classifier_acc={lgt_acc} content_lpips={lgt_lp} art_fid={lgt_artfid}")
        print(f"[cmp][full_eval] SD1.5  classifier_acc={sd_acc} content_lpips={sd_lp} art_fid={sd_artfid}")

    if str(args.out_json).strip():
        out = Path(args.out_json).resolve()
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"[cmp] saved: {out}")


if __name__ == "__main__":
    main()
