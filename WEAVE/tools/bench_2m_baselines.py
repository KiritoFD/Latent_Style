"""Precise wall-clock benchmark for the two "2m" baselines in the paper:
SD-Turbo (L318, claimed 2m) and StyleAligned (L319, claimed 2m).

Unified timing protocol (one model load, then per-image split timing):
  - load_time      : model + VAE load (excludes nothing)
  - inversion_time : StyleAligned only, one-time DDIM inversion of 5 refs
  - sum_pipe       : pure model inference across all pairs (excludes disk I/O)
  - sum_save       : PNG encode + disk write across all pairs
  - pipe_per_img   : sum_pipe / n
Reported totals:
  - total_nosave   = load + inversion + sum_pipe
  - total_withsave = load + inversion + sum_pipe + sum_save
These let us compare apples-to-apples with WEAVE's 83.7s and the paper's 120s.
"""
import sys
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.stderr.reconfigure(encoding="utf-8", errors="replace")
import argparse
import json
import time
from pathlib import Path

import torch
from PIL import Image

DEVICE = "cuda"
IMG = 512
STYLES = ["Early_Renaissance", "Impressionism", "Minimalism", "Rococo", "Ukiyo_e"]
STYLE_PROMPT = {
    "Early_Renaissance": "early renaissance painting",
    "Impressionism": "impressionist painting",
    "Minimalism": "minimalist painting",
    "Rococo": "rococo painting",
    "Ukiyo_e": "ukiyo-e painting",
}
MANIFEST = Path("G:/GitHub/Latent_Style/Dataset/distinct5_512/test_manifest.json")


def build_pairs(max_src=None, max_pairs=None):
    manifest = json.load(open(MANIFEST, encoding="utf-8"))
    test_dir = Path(manifest["test_dir"])
    style_files = manifest["style_files"]
    pairs = []
    src_styles = STYLES[: max_src] if max_src else STYLES
    for src in src_styles:
        for tgt in STYLES:
            for fn in style_files[src]:
                pairs.append((src, Path(fn).stem, tgt))
                if max_pairs and len(pairs) >= max_pairs:
                    return test_dir, pairs
    return test_dir, pairs


def find_src(test_dir, src_style, src_stem):
    for ext in (".jpg", ".jpeg", ".png"):
        p = test_dir / src_style / (src_stem + ext)
        if p.exists():
            return p
    return None


# ---------------------------------------------------------------------------
def bench_sdturbo(pairs, test_dir, out_root):
    from diffusers import StableDiffusionImg2ImgPipeline

    out_dir = out_root / "bench_sdturbo"
    out_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
        "stabilityai/sd-turbo",
        torch_dtype=torch.float16,
        safety_checker=None,
        requires_safety_checker=False,
    ).to(DEVICE)
    load_time = time.time() - t0

    sum_pipe = 0.0
    sum_save = 0.0
    n = 0
    for src_style, src_stem, tgt in pairs:
        sp = find_src(test_dir, src_style, src_stem)
        if sp is None:
            continue
        img = Image.open(sp).convert("RGB").resize((IMG, IMG), Image.LANCZOS)
        gen = torch.Generator(device="cpu").manual_seed(1234 + n)
        tp = time.time()
        res = pipe(
            prompt=STYLE_PROMPT[tgt],
            image=img,
            strength=0.8,
            num_inference_steps=2,
            guidance_scale=0.0,
            generator=gen,
        )
        sum_pipe += time.time() - tp
        out_path = out_dir / f"{src_style}__{src_stem}__to__{tgt}.png"
        ts = time.time()
        res.images[0].save(out_path)
        sum_save += time.time() - ts
        n += 1
    return {
        "method": "SD-Turbo",
        "n_images": n,
        "steps": 2,
        "load_time": load_time,
        "inversion_time": 0.0,
        "sum_pipe": sum_pipe,
        "sum_save": sum_save,
        "pipe_per_img": sum_pipe / n if n else None,
        "save_per_img": sum_save / n if n else None,
        "total_nosave": load_time + sum_pipe,
        "total_withsave": load_time + sum_pipe + sum_save,
    }


# ---------------------------------------------------------------------------
def bench_stylealigned(pairs, test_dir, out_root):
    sys.path.insert(0, str(Path(__file__).resolve().parent / "style_aligned"))
    from sa_handler_sd15 import Handler, StyleAlignedArgs
    import inversion_sd15 as inversion
    from diffusers import StableDiffusionPipeline, DDIMScheduler

    out_dir = out_root / "bench_stylealigned"
    out_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    pipe = StableDiffusionPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        torch_dtype=torch.float16,
        safety_checker=None,
        requires_safety_checker=False,
    ).to(DEVICE)
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    load_time = time.time() - t0

    handler = Handler(pipe)
    handler.register(StyleAlignedArgs(
        share_group_norm=True, share_layer_norm=True, share_attention=True,
        adain_queries=True, adain_keys=True, adain_values=False,
        shared_score_shift=0.0, shared_score_scale=1.0, only_self_level=0.0,
    ))

    # One-time DDIM inversion of 5 style references
    ti0 = time.time()
    style_inversions = {}
    for style in STYLES:
        cands = sorted([p for p in (test_dir / style).iterdir()
                        if p.suffix.lower() in {".jpg", ".jpeg", ".png"}])
        ref = Image.open(cands[0]).convert("RGB").resize((IMG, IMG), Image.LANCZOS)
        zts = inversion.ddim_inversion(
            pipe, ref, f"a {STYLE_PROMPT[style]}",
            num_inference_steps=20, guidance_scale=3.5,
        )
        style_inversions[style] = zts
    inversion_time = time.time() - ti0

    sum_pipe = 0.0
    sum_save = 0.0
    n = 0
    for src_style, src_stem, tgt in pairs:
        sp = find_src(test_dir, src_style, src_stem)
        if sp is None:
            continue
        img = Image.open(sp).convert("RGB").resize((IMG, IMG), Image.LANCZOS)
        zts = style_inversions[tgt]
        zT, cb = inversion.make_inversion_callback(zts, offset=0)
        latents = torch.randn(2, 4, 64, 64, device="cpu",
                              generator=torch.Generator(device="cpu").manual_seed(42),
                              dtype=pipe.unet.dtype).to(DEVICE)
        latents[0] = zT
        tp = time.time()
        images = pipe(
            [f"a {STYLE_PROMPT[tgt]}", f"a {STYLE_PROMPT[tgt]}"],
            latents=latents,
            callback_on_step_end=cb,
            num_inference_steps=20,
            guidance_scale=7.5,
        ).images
        sum_pipe += time.time() - tp
        out_path = out_dir / f"{src_style}__{src_stem}__to__{tgt}.png"
        ts = time.time()
        images[1].save(out_path)
        sum_save += time.time() - ts
        n += 1
    handler.remove()
    return {
        "method": "StyleAligned",
        "n_images": n,
        "steps": 20,
        "load_time": load_time,
        "inversion_time": inversion_time,
        "sum_pipe": sum_pipe,
        "sum_save": sum_save,
        "pipe_per_img": sum_pipe / n if n else None,
        "save_per_img": sum_save / n if n else None,
        "total_nosave": load_time + inversion_time + sum_pipe,
        "total_withsave": load_time + inversion_time + sum_pipe + sum_save,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", choices=["sdturbo", "stylealigned", "all"], default="all")
    ap.add_argument("--max_src", type=int, default=None,
                    help="Limit #source styles (for fast subset timing).")
    ap.add_argument("--max_pairs", type=int, default=None,
                    help="Hard cap on #pairs (for a tiny timing subset).")
    ap.add_argument("--out", default="exp/bench_2m")
    args = ap.parse_args()

    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)
    test_dir, pairs = build_pairs(args.max_src, args.max_pairs)
    print(f"[bench] method={args.method} max_src={args.max_src} pairs={len(pairs)}")

    results = []
    if args.method in ("sdturbo", "all"):
        r = bench_sdturbo(pairs, test_dir, out_root)
        results.append(r)
        print(json.dumps(r, indent=2))
    if args.method in ("stylealigned", "all"):
        r = bench_stylealigned(pairs, test_dir, out_root)
        results.append(r)
        print(json.dumps(r, indent=2))

    (out_root / "bench_results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"[bench] saved -> {out_root / 'bench_results.json'}")


if __name__ == "__main__":
    main()
