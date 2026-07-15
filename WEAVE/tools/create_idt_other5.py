"""Create IDT (identity) image directory for other5 evaluation.

For each (src_style, src_stem, tgt_style) triplet, creates a hardlink
from the source image to {src_style}__{src_stem}__to__{tgt_style}.png.
No style transfer is applied — the "generated" image is the source itself.
"""
import argparse
import os
from pathlib import Path

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
STYLE_NAMES_DEFAULT = [
    "Abstract_Expressionism",
    "Art_Nouveau_Modern",
    "Cubism",
    "Expressionism",
    "Symbolism",
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--style-names", default=",".join(STYLE_NAMES_DEFAULT))
    parser.add_argument("--num-src", type=int, default=30)
    args = parser.parse_args()

    style_names = [s.strip() for s in args.style_names.split(",") if s.strip()]
    test_root = Path(args.test_root)
    out_dir = Path(args.output_root) / "step_000001" / "images"
    out_dir.mkdir(parents=True, exist_ok=True)

    total = 0
    for src_style in style_names:
        src_dir = test_root / src_style
        if not src_dir.exists():
            continue
        src_files = sorted([f for f in src_dir.iterdir() if f.suffix.lower() in IMAGE_EXTS])[:args.num_src]
        for src_path in src_files:
            for tgt_style in style_names:
                out_name = f"{src_style}__{src_path.stem}__to__{tgt_style}.png"
                out_path = out_dir / out_name
                if out_path.exists():
                    continue
                # Use hardlink (same volume), fallback to copy
                try:
                    os.link(src_path, out_path)
                except OSError:
                    import shutil
                    shutil.copy2(src_path, out_path)
                total += 1
    print(f"[INFO] Created {total} IDT images in {out_dir}")


if __name__ == "__main__":
    main()
