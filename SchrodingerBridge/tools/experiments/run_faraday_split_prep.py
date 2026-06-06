from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
SB_ROOT = SCRIPT_DIR.parent.parent
WORKSPACE_ROOT = SB_ROOT.parent


def _run(cmd: list[str]) -> None:
    print("RUN:", " ".join(cmd), flush=True)
    completed = subprocess.run(cmd, cwd=WORKSPACE_ROOT, check=False)
    if completed.returncode != 0:
        raise SystemExit(completed.returncode)


def _style_list(raw: str) -> list[str]:
    styles = [item.strip() for item in str(raw).split(",") if item.strip()]
    if not styles:
        raise ValueError("styles must not be empty")
    return styles


def _count_latents(root: Path, styles: list[str]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for style in styles:
        style_dir = root / style
        counts[style] = len(sorted(style_dir.glob("*.pt"))) if style_dir.is_dir() else 0
    return counts


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Prepare a remote fixed-rule WikiArt stress split for LBM training by encoding train images, building packed latent cache, and building prototype pairing cache."
    )
    parser.add_argument("--split-root", required=True, help="Remote split root, for example /mnt/i/wikiart_faraday_splits/wikiart_stress1_...")
    parser.add_argument("--styles", required=True, help="Comma-separated style names.")
    parser.add_argument("--python-bin", default=sys.executable)
    parser.add_argument("--vae-model", default="ema")
    parser.add_argument("--vae-cache-dir", default="/mnt/i/Github/Latent_Style/eval_cache/hf")
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260603)
    parser.add_argument("--overwrite-latents", action="store_true")
    parser.add_argument("--rebuild-cache", action="store_true")
    args = parser.parse_args()

    styles = _style_list(args.styles)
    split_root = Path(args.split_root).resolve()
    classview_train = split_root / "classview" / "train"
    classview_test = split_root / "classview" / "test"
    latents_train = split_root / "latents_ema" / "train"
    pairing_cache = latents_train / ".latent_cache" / "prototype_pairing_top8.pt"
    packed_manifest = latents_train / ".latent_cache" / "manifest.json"

    if not classview_train.is_dir():
        raise FileNotFoundError(f"Missing classview train root: {classview_train}")
    if not classview_test.is_dir():
        raise FileNotFoundError(f"Missing classview test root: {classview_test}")

    encode_cmd = [
        str(args.python_bin),
        "SchrodingerBridge/tools/encode_image_folder_latents.py",
        "--input-root",
        str(classview_train),
        "--output-root",
        str(latents_train),
        "--vae-model",
        str(args.vae_model),
        "--vae-cache-dir",
        str(args.vae_cache_dir),
        "--image-size",
        str(int(args.image_size)),
        "--batch-size",
        str(int(args.batch_size)),
        "--latent-mode",
        "mode",
        "--class-list",
        *styles,
        "--device",
        "cuda",
        "--seed",
        str(int(args.seed)),
    ]
    if args.overwrite_latents:
        encode_cmd.append("--overwrite")

    _run(encode_cmd)

    if args.rebuild_cache or not packed_manifest.exists():
        _run(
            [
                str(args.python_bin),
                "SchrodingerBridge/tools/build_latent_packed_cache.py",
                "--data-root",
                str(latents_train),
                "--styles",
                ",".join(styles),
            ]
        )

    if args.rebuild_cache or not pairing_cache.exists():
        _run(
            [
                str(args.python_bin),
                "SchrodingerBridge/tools/build_latent_prototype_pairing_cache.py",
                "--data-root",
                str(latents_train),
                "--styles",
                ",".join(styles),
                "--output",
                str(pairing_cache),
                "--topk",
                "8",
                "--num-prototypes",
                "8",
                "--pool-size",
                "4",
                "--chunk-size",
                "128",
                "--cross-only",
            ]
        )

    payload = {
        "split_root": str(split_root),
        "styles": styles,
        "latents_train": str(latents_train),
        "counts": _count_latents(latents_train, styles),
        "packed_manifest": str(packed_manifest),
        "pairing_cache": str(pairing_cache),
    }
    print(json.dumps(payload, indent=2, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
