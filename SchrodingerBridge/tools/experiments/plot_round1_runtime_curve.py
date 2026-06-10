from __future__ import annotations

import argparse
import csv
import json
import math
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _safe_float(value: Any) -> float | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def _load_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            text = line.strip()
            if not text:
                continue
            rows.append(json.loads(text))
    return rows


def _progress(row: dict[str, Any]) -> float:
    epoch = _safe_float(row.get("remote_live_epoch")) or 0.0
    epoch_total = _safe_float(row.get("remote_live_epoch_total")) or 1.0
    step = _safe_float(row.get("remote_live_step")) or 0.0
    step_total = _safe_float(row.get("remote_live_step_total")) or 1.0
    if epoch_total <= 0 or step_total <= 0:
        return 0.0
    return (epoch - 1.0) + (step / step_total)


def _timestamp_seconds(row: dict[str, Any], start_dt: datetime) -> float:
    dt = datetime.fromisoformat(str(row["timestamp"]))
    return (dt - start_dt).total_seconds() / 60.0


def _plot_value(value: Any) -> float:
    parsed = _safe_float(value)
    return float(parsed) if parsed is not None else float("nan")


def main() -> int:
    parser = argparse.ArgumentParser(description="Plot remote round-1 runtime curves from runtime_samples.jsonl.")
    parser.add_argument("--input-jsonl", type=Path, required=True)
    parser.add_argument("--output-png", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, default=None)
    args = parser.parse_args()

    input_jsonl = Path(args.input_jsonl).resolve()
    rows = _load_rows(input_jsonl)
    if not rows:
        raise RuntimeError(f"No runtime rows found in {input_jsonl}")

    start_dt = datetime.fromisoformat(str(rows[0]["timestamp"]))
    table_rows: list[dict[str, Any]] = []
    for idx, row in enumerate(rows):
        table_rows.append(
            {
                "index": idx,
                "timestamp": row.get("timestamp", ""),
                "minutes_from_start": _timestamp_seconds(row, start_dt),
                "progress_epoch": _progress(row),
                "epoch": _safe_float(row.get("remote_live_epoch")),
                "epoch_total": _safe_float(row.get("remote_live_epoch_total")),
                "step": _safe_float(row.get("remote_live_step")),
                "step_total": _safe_float(row.get("remote_live_step_total")),
                "loss": _safe_float(row.get("remote_live_loss")),
                "tswd": _safe_float(row.get("remote_live_tswd")),
                "memory_used_mib": _safe_float(row.get("remote_live_memory_used_mib")),
                "memory_total_mib": _safe_float(row.get("remote_live_memory_total_mib")),
                "util_pct": _safe_float(row.get("remote_live_util_pct")),
                "band_status": row.get("remote_live_band_status", ""),
                "formal_status": row.get("remote_live_formal_status", ""),
            }
        )

    out_csv = Path(args.output_csv).resolve() if args.output_csv is not None else input_jsonl.with_suffix(".csv")
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(table_rows[0].keys()))
        writer.writeheader()
        writer.writerows(table_rows)

    xs = [float(row["progress_epoch"]) for row in table_rows]
    minutes = [float(row["minutes_from_start"]) for row in table_rows]
    losses = [_plot_value(row["loss"]) for row in table_rows]
    tswds = [_plot_value(row["tswd"]) for row in table_rows]
    mems = [_plot_value(row["memory_used_mib"]) for row in table_rows]
    utils = [_plot_value(row["util_pct"]) for row in table_rows]

    fig, axes = plt.subplots(2, 1, figsize=(8.2, 6.3), dpi=180, sharex=True)

    axes[0].plot(xs, losses, marker="o", linewidth=1.6, color="#C0392B", label="loss")
    axes[0].set_ylabel("loss", color="#C0392B")
    axes[0].tick_params(axis="y", labelcolor="#C0392B")
    axes[0].grid(True, alpha=0.25)
    ax0b = axes[0].twinx()
    ax0b.plot(xs, tswds, marker="s", linewidth=1.4, color="#1D4ED8", label="tswd")
    ax0b.set_ylabel("tswd", color="#1D4ED8")
    ax0b.tick_params(axis="y", labelcolor="#1D4ED8")
    axes[0].set_title("Round-1 remote runtime curve")

    axes[1].plot(xs, mems, marker="o", linewidth=1.6, color="#16A085", label="memory_used_mib")
    axes[1].set_ylabel("VRAM MiB", color="#16A085")
    axes[1].tick_params(axis="y", labelcolor="#16A085")
    axes[1].grid(True, alpha=0.25)
    ax1b = axes[1].twinx()
    ax1b.plot(xs, utils, marker="s", linewidth=1.4, color="#7C3AED", label="util_pct")
    ax1b.set_ylabel("util %", color="#7C3AED")
    ax1b.tick_params(axis="y", labelcolor="#7C3AED")
    axes[1].axhspan(9216, 11059, color="#F3E7D6", alpha=0.38, zorder=0)
    axes[1].axhline(11571, color="#B91C1C", linewidth=1.1, linestyle="--")
    axes[1].set_xlabel("epoch-progress")

    if minutes:
        for idx in range(0, len(table_rows), max(1, len(table_rows) // 6)):
            if math.isnan(mems[idx]):
                continue
            axes[1].annotate(
                f"{minutes[idx]:.0f}m",
                (xs[idx], mems[idx]),
                xytext=(4, 4),
                textcoords="offset points",
                fontsize=7,
                color="#4B5563",
            )

    fig.tight_layout()
    out_png = Path(args.output_png).resolve()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png)
    plt.close(fig)
    print(out_png)
    print(out_csv)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
