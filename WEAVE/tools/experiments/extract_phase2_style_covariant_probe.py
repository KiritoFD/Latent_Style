from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path


FIELDS = [
    "config_id",
    "family",
    "sigma",
    "transfer_clip_style",
    "transfer_lpips",
    "wall_total",
    "lancet_generation",
    "vae_decode",
    "eval_total",
    "i2sb_style_noise_family_style_covariant",
    "i2sb_style_noise_family_gaussian",
    "i2sb_style_noise_bank_active",
    "i2sb_style_noise_amp_mean",
    "i2sb_style_noise_amp_std",
    "i2sb_style_noise_post_std",
    "i2sb_style_noise_amplitude_power",
    "i2sb_style_noise_fallback_gaussian",
]


def _infer_family(config_id: str) -> str:
    if "stylecov" in config_id:
        return "style_covariant"
    if "gaussian" in config_id:
        return "gaussian"
    return "deterministic"


def _infer_sigma(config_id: str) -> str:
    for token in ("sigma0p0", "sigma0p5", "sigma0p8", "sigma1p2"):
        if token in config_id:
            return token.replace("sigma", "").replace("p", ".")
    return ""


def _read_summary(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _extract_row(summary_path: Path) -> dict[str, object]:
    summary = _read_summary(summary_path)
    config_id = summary_path.parent.parent.name
    transfer = (((summary.get("analysis") or {}).get("style_transfer_ability")) or {})
    runtime = (((summary.get("runtime_observability") or {}).get("style_transfer_ability")) or {})
    timings = (summary.get("timings_sec") or {})
    row: dict[str, object] = {
        "config_id": config_id,
        "family": _infer_family(config_id),
        "sigma": _infer_sigma(config_id),
        "transfer_clip_style": transfer.get("clip_style"),
        "transfer_lpips": transfer.get("content_lpips"),
        "wall_total": timings.get("wall_total"),
        "lancet_generation": timings.get("lancet_generation"),
        "vae_decode": timings.get("vae_decode"),
        "eval_total": timings.get("eval_total"),
    }
    for key in FIELDS:
        row.setdefault(key, runtime.get(key))
    return row


def main() -> int:
    parser = argparse.ArgumentParser(description="Extract phase2 style-covariant probe summaries into a flat CSV.")
    parser.add_argument("--root", required=True, help="Root directory that contains config_id/epoch_xxxx/summary.json outputs.")
    parser.add_argument("--output", default="", help="Optional CSV output path. Defaults to stdout.")
    args = parser.parse_args()

    root = Path(args.root).expanduser().resolve()
    rows = [_extract_row(path) for path in sorted(root.glob("*/epoch_0009/summary.json"))]

    if str(args.output).strip():
        out_path = Path(args.output).expanduser().resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_fp = out_path.open("w", encoding="utf-8", newline="")
        close_fp = True
    else:
        out_fp = sys.stdout
        close_fp = False

    try:
        writer = csv.DictWriter(out_fp, fieldnames=FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    finally:
        if close_fp:
            out_fp.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
