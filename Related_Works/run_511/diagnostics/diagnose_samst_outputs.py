"""Visual and numeric diagnostics for SaMST outputs."""
from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


WORKSPACE_ROOT = Path(__file__).resolve().parent.parent
STYLE_ROOT = WORKSPACE_ROOT / "style_data" / "overfit50"

SAMPLES = [
    "photo_2013-11-08 16_45_24_to_vangogh.jpg",
    "photo_2013-11-08 16_45_24_to_Hayao.jpg",
    "cezanne_00057_to_photo.jpg",
    "Hayao_0_to_monet.jpg",
]

RUNS = [
    ("SRC", None),
    ("Ours e7", WORKSPACE_ROOT / "SchrodingerBridge" / "S-add__K-1_C-0_W-20_Col-0" / "full_eval" / "epoch_0007" / "images"),
    ("SaMST", WORKSPACE_ROOT / "run_511" / "outputs" / "samst_750_strict" / "infer_750" / "images"),
    ("AdaIN", WORKSPACE_ROOT / "run_511" / "outputs" / "adain_7g_v32k" / "infer_750" / "images"),
    ("StyleID", WORKSPACE_ROOT / "run_511" / "outputs" / "styleid_750_strict" / "infer_750" / "images"),
]


def source_path(name: str) -> Path:
    prefix, _target = name[:-4].rsplit("_to_", 1)
    src_style, stem = prefix.split("_", 1)
    return STYLE_ROOT / src_style / f"{stem}.jpg"


def image_stats(path: Path) -> dict[str, object]:
    arr = np.asarray(Image.open(path).convert("RGB")).astype("float32") / 255.0
    high_freq = float(np.abs(np.diff(arr, axis=0)).mean() + np.abs(np.diff(arr, axis=1)).mean())
    return {
        "mean": arr.mean(axis=(0, 1)).round(3).tolist(),
        "std": arr.std(axis=(0, 1)).round(3).tolist(),
        "min": round(float(arr.min()), 3),
        "max": round(float(arr.max()), 3),
        "hf": round(high_freq, 4),
    }


def make_contact_sheet(out_path: Path) -> None:
    thumb = 180
    label_h = 32
    width = thumb * len(RUNS)
    height = (thumb + label_h) * len(SAMPLES)
    sheet = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(sheet)

    for row, name in enumerate(SAMPLES):
        for col, (label, root) in enumerate(RUNS):
            x = col * thumb
            y = row * (thumb + label_h)
            draw.text((x + 4, y + 4), label, fill=(0, 0, 0))
            path = source_path(name) if root is None else root / name
            if path.exists():
                img = Image.open(path).convert("RGB").resize((thumb, thumb))
                sheet.paste(img, (x, y + label_h))
            else:
                draw.text((x + 4, y + label_h + 40), "MISSING", fill=(255, 0, 0))
    sheet.save(out_path, quality=95)


def main() -> int:
    out_img = WORKSPACE_ROOT / "run_511" / "diagnostic_samst_contact.jpg"
    out_md = WORKSPACE_ROOT / "run_511" / "diagnostic_samst_stats.md"
    make_contact_sheet(out_img)

    lines = [
        "# SaMST Diagnostic Stats",
        "",
        f"Contact sheet: `{out_img}`",
        "",
        "| Image | Run | RGB mean | RGB std | min | max | high-freq |",
        "| --- | --- | --- | --- | ---: | ---: | ---: |",
    ]
    for name in SAMPLES:
        for label, root in RUNS:
            path = source_path(name) if root is None else root / name
            if not path.exists():
                continue
            st = image_stats(path)
            lines.append(
                f"| `{name}` | {label} | {st['mean']} | {st['std']} | "
                f"{st['min']} | {st['max']} | {st['hf']} |"
            )
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(out_img)
    print(out_md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
