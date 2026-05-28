from __future__ import annotations

import argparse
import csv
from pathlib import Path


EMA_REFERENCES = {
    "mse_plain4_w20_anchor": {
        "ema_variant": "ema_plain4_w20_anchor",
        "epoch": 6,
        "clip_style": 0.7007,
        "content_lpips": 0.4215,
        "source": "docs/experiments/2026-05-26-klf4-ema-vae-backend.md",
    },
    "mse_dynamic_guard_w28": {
        "ema_variant": "ema_dynamic_guard_w28",
        "epoch": 6,
        "clip_style": 0.7078,
        "content_lpips": 0.4477,
        "source": "docs/experiments/2026-05-26-klf4-ema-vae-backend.md",
    },
    "mse_transport_texton_w34_guard": {
        "ema_variant": "ema_transport_texton_w34_guard",
        "epoch": 6,
        "clip_style": 0.71451,
        "content_lpips": 0.48261,
        "source": "docs/experiments/2026-05-27-seedream-teacher-adapter-and-ema-mainline.md",
    },
    "mse_bodyblend_w28_guard": {
        "ema_variant": "ema_bodyblend_w28_guard",
        "epoch": 6,
        "clip_style": 0.7158,
        "content_lpips": 0.4972,
        "source": "docs/logs/experiment_ledger.md",
    },
    "mse_guard_w20_lowwarp": {
        "ema_variant": "ema_guard_w20_lowwarp",
        "epoch": 7,
        "clip_style": 0.7245,
        "content_lpips": 0.5526,
        "source": "docs/logs/experiment_ledger.md",
    },
}


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _float(row: dict[str, str], key: str) -> float:
    value = row.get(key, "")
    return float(value) if value not in {"", None} else float("nan")


def _best_rows(rows: list[dict[str, str]]) -> dict[str, dict[str, str]]:
    best: dict[str, dict[str, str]] = {}
    for row in rows:
        if row.get("status") != "ok":
            continue
        variant = row["variant"]
        current = best.get(variant)
        if current is None or _float(row, "clip_style") > _float(current, "clip_style"):
            best[variant] = row
    return best


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _write_md(path: Path, rows: list[dict[str, object]]) -> None:
    lines = [
        "# MSE Backend Control Comparison",
        "",
        "Matched controls clone the current EMA variant and change only the VAE backend to `mse` plus `latent-256`.",
        "",
        "| family | EMA clip | EMA LPIPS | MSE best epoch | MSE clip | MSE LPIPS | d clip | d LPIPS | readout |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        lines.append(
            "| {family} | {ema_clip_style:.6f} | {ema_content_lpips:.6f} | {mse_epoch} | "
            "{mse_clip_style:.6f} | {mse_content_lpips:.6f} | {delta_clip_style:+.6f} | "
            "{delta_content_lpips:+.6f} | {readout} |".format(**row)
        )
    lines.extend(
        [
            "",
            "Decision rule:",
            "",
            "- MSE is promoted only if it raises style clearly while preserving the LPIPS budget.",
            "- A small style gain below `0.72` is not enough to pivot the mainline away from EMA.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mse-csv", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    best = _best_rows(_read_rows(args.mse_csv))
    comparison: list[dict[str, object]] = []
    for mse_variant, ema in EMA_REFERENCES.items():
        row = best.get(mse_variant)
        if row is None:
            continue
        mse_clip = _float(row, "clip_style")
        mse_lpips = _float(row, "content_lpips")
        ema_clip = float(ema["clip_style"])
        ema_lpips = float(ema["content_lpips"])
        delta_clip = mse_clip - ema_clip
        delta_lpips = mse_lpips - ema_lpips
        if mse_clip >= 0.72 and mse_lpips <= 0.50:
            readout = "promising"
        elif delta_clip > 0.002 and mse_lpips <= ema_lpips + 0.01:
            readout = "small_gain"
        elif mse_lpips < ema_lpips - 0.03 and mse_clip < ema_clip:
            readout = "content_only"
        else:
            readout = "negative_or_neutral"
        comparison.append(
            {
                "family": str(ema["ema_variant"]).replace("ema_", ""),
                "ema_variant": ema["ema_variant"],
                "ema_epoch": ema["epoch"],
                "ema_clip_style": ema_clip,
                "ema_content_lpips": ema_lpips,
                "mse_variant": mse_variant,
                "mse_epoch": int(row["epoch"]),
                "mse_clip_style": mse_clip,
                "mse_content_lpips": mse_lpips,
                "delta_clip_style": delta_clip,
                "delta_content_lpips": delta_lpips,
                "readout": readout,
                "ema_source": ema["source"],
                "mse_summary": row.get("summary", ""),
            }
        )

    _write_csv(args.out_dir / "mse_backend_matched_comparison.csv", comparison)
    _write_md(args.out_dir / "mse_backend_matched_comparison.md", comparison)
    print(args.out_dir.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
