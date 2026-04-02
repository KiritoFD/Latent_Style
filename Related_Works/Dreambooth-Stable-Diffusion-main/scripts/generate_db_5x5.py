import argparse
import hashlib
import shutil
import subprocess
import sys
from pathlib import Path


def list_image_files(folder: Path):
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    return sorted([p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in exts])


def parse_styles(raw: str):
    return [x.strip() for x in str(raw).split(",") if x.strip()]


def deterministic_seed(base_seed: int, src_style: str, src_stem: str, tgt_style: str) -> int:
    key = f"{base_seed}|{src_style}|{src_stem}|{tgt_style}".encode("utf-8")
    h = hashlib.md5(key).hexdigest()
    return int(h[:8], 16) % (2**31 - 1)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--repo_root", type=str, required=True)
    p.add_argument("--ckpt", type=str, required=True)
    p.add_argument("--out_dir", type=str, required=True)
    p.add_argument("--test_dir", type=str, required=True)
    p.add_argument("--styles", type=str, default="photo,cezanne,Hayao,monet,vangogh")
    p.add_argument("--identifier", type=str, default="sks")
    p.add_argument("--class_word", type=str, default="object")
    p.add_argument("--ddim_steps", type=int, default=50)
    p.add_argument("--scale", type=float, default=7.5)
    p.add_argument("--H", type=int, default=512)
    p.add_argument("--W", type=int, default=512)
    p.add_argument("--base_seed", type=int, default=42)
    args = p.parse_args()

    repo_root = Path(args.repo_root).resolve()
    ckpt = Path(args.ckpt).resolve()
    test_dir = Path(args.test_dir).resolve()
    out_dir = Path(args.out_dir).resolve()
    images_dir = out_dir / "images"
    tmp_dir = out_dir / "_tmp_samples"
    images_dir.mkdir(parents=True, exist_ok=True)
    tmp_dir.mkdir(parents=True, exist_ok=True)

    styles = parse_styles(args.styles)
    if not styles:
        raise ValueError("No styles provided.")

    src_pick = {}
    for s in styles:
        sdir = test_dir / s
        if not sdir.exists():
            raise FileNotFoundError(f"Missing style dir: {sdir}")
        files = list_image_files(sdir)
        if not files:
            raise FileNotFoundError(f"No images found in: {sdir}")
        src_pick[s] = files[0]

    for src_style in styles:
        src_path = src_pick[src_style]
        src_stem = src_path.stem
        for tgt_style in styles:
            seed = deterministic_seed(args.base_seed, src_style, src_stem, tgt_style)
            prompt = f"a photo of a {args.identifier} {args.class_word} in the style of {tgt_style}"

            cmd = [
                sys.executable,
                str(repo_root / "scripts" / "stable_txt2img.py"),
                "--prompt",
                prompt,
                "--skip_grid",
                "--n_samples",
                "1",
                "--n_iter",
                "1",
                "--ddim_steps",
                str(args.ddim_steps),
                "--scale",
                str(args.scale),
                "--ddim_eta",
                "0.0",
                "--seed",
                str(seed),
                "--H",
                str(args.H),
                "--W",
                str(args.W),
                "--config",
                str(repo_root / "configs" / "stable-diffusion" / "v1-inference.yaml"),
                "--ckpt",
                str(ckpt),
                "--outdir",
                str(tmp_dir),
            ]
            subprocess.run(cmd, check=True, cwd=str(repo_root))

            samples = list_image_files(tmp_dir / "samples")
            if not samples:
                raise RuntimeError("No sample generated in tmp samples folder.")
            latest = max(samples, key=lambda x: x.stat().st_mtime)

            out_name = f"{src_style}_{src_stem}_to_{tgt_style}.jpg"
            out_path = images_dir / out_name
            shutil.copy2(latest, out_path)
            print(out_path)

    print(f"done: {images_dir}")


if __name__ == "__main__":
    main()

