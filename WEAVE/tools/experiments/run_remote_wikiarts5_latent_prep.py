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
    print("RUN:", " ".join(str(x) for x in cmd), flush=True)
    completed = subprocess.run(cmd, cwd=WORKSPACE_ROOT, check=False)
    if completed.returncode != 0:
        raise SystemExit(completed.returncode)


def _count_latents(root: Path, styles: list[str]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for style in styles:
        style_dir = root / style
        counts[style] = len(sorted(style_dir.glob("*.pt"))) if style_dir.is_dir() else 0
    return counts


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare remote train latents + packed cache + pairing cache for the new full wikiarts-5 RGB dataset.")
    parser.add_argument("--image-root", required=True)
    parser.add_argument("--latent-root", required=True)
    parser.add_argument("--styles", required=True)
    parser.add_argument("--python-bin", default=sys.executable)
    parser.add_argument("--vae-model", default="ema")
    parser.add_argument("--vae-cache-dir", default="/mnt/i/Github/Latent_Style/eval_cache/hf")
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260610)
    parser.add_argument("--overwrite-latents", action="store_true")
    parser.add_argument("--rebuild-cache", action="store_true")
    args = parser.parse_args()

    styles = [x.strip() for x in str(args.styles).split(",") if x.strip()]
    image_root = Path(args.image_root).resolve()
    latent_root = Path(args.latent_root).resolve()
    pairing_cache = latent_root / ".latent_cache" / "prototype_pairing_top8.pt"
    packed_manifest = latent_root / ".latent_cache" / "manifest.json"

    if not image_root.is_dir():
        raise FileNotFoundError(f"Missing RGB train root: {image_root}")

    encode_cmd = [
        str(args.python_bin),
        "SchrodingerBridge/tools/encode_image_folder_latents.py",
        "--input-root",
        str(image_root),
        "--output-root",
        str(latent_root),
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
    if bool(args.overwrite_latents):
        encode_cmd.append("--overwrite")
    _run(encode_cmd)

    if bool(args.rebuild_cache) or not packed_manifest.exists():
        _run(
            [
                str(args.python_bin),
                "SchrodingerBridge/tools/build_latent_packed_cache.py",
                "--data-root",
                str(latent_root),
                "--styles",
                ",".join(styles),
            ]
        )

    if bool(args.rebuild_cache) or not pairing_cache.exists():
        _run(
            [
                str(args.python_bin),
                "SchrodingerBridge/tools/build_latent_prototype_pairing_cache.py",
                "--data-root",
                str(latent_root),
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
        "image_root": str(image_root),
        "latent_root": str(latent_root),
        "styles": styles,
        "counts": _count_latents(latent_root, styles),
        "packed_manifest": str(packed_manifest),
        "pairing_cache": str(pairing_cache),
    }
    print(json.dumps(payload, indent=2, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
