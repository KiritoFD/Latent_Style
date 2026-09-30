from __future__ import annotations

import argparse
import csv
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


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def main() -> int:
    parser = argparse.ArgumentParser(description="Plot a round-1 remote training CSV into a compact summary figure.")
    parser.add_argument("--input-csv", type=Path, required=True)
    parser.add_argument("--output-png", type=Path, required=True)
    args = parser.parse_args()

    rows = _read_rows(Path(args.input_csv).resolve())
    if not rows:
        raise RuntimeError(f"Empty training csv: {args.input_csv}")

    epochs = [_safe_float(row.get("epoch")) for row in rows]
    losses = [_safe_float(row.get("loss")) for row in rows]
    terminal_swd = [_safe_float(row.get("terminal_swd")) for row in rows]
    samples_per_sec = [_safe_float(row.get("samples_per_sec")) for row in rows]
    cuda_alloc = [_safe_float(row.get("cuda_peak_allocated_gb")) for row in rows]
    cuda_reserved = [_safe_float(row.get("cuda_peak_reserved_gb")) for row in rows]

    fig, axes = plt.subplots(2, 1, figsize=(8.0, 6.0), dpi=180, sharex=True)

    axes[0].plot(epochs, losses, marker="o", linewidth=1.8, color="#C0392B", label="loss")
    axes[0].set_ylabel("loss", color="#C0392B")
    axes[0].tick_params(axis="y", labelcolor="#C0392B")
    axes[0].grid(True, alpha=0.25)
    ax0b = axes[0].twinx()
    ax0b.plot(epochs, terminal_swd, marker="s", linewidth=1.4, color="#1D4ED8", label="terminal_swd")
    ax0b.set_ylabel("terminal_swd", color="#1D4ED8")
    ax0b.tick_params(axis="y", labelcolor="#1D4ED8")
    axes[0].set_title("Round-1 remote training curve")

    axes[1].plot(epochs, samples_per_sec, marker="o", linewidth=1.7, color="#16A085", label="samples_per_sec")
    axes[1].set_ylabel("samples/sec", color="#16A085")
    axes[1].tick_params(axis="y", labelcolor="#16A085")
    axes[1].grid(True, alpha=0.25)
    ax1b = axes[1].twinx()
    ax1b.plot(epochs, cuda_alloc, marker="s", linewidth=1.2, color="#7C3AED", label="cuda_peak_allocated_gb")
    ax1b.plot(epochs, cuda_reserved, marker="^", linewidth=1.2, color="#B45309", label="cuda_peak_reserved_gb")
    ax1b.set_ylabel("CUDA peak GB", color="#7C3AED")
    ax1b.tick_params(axis="y", labelcolor="#7C3AED")
    axes[1].set_xlabel("epoch")

    for idx, epoch in enumerate(epochs):
        if epoch is None:
            continue
        if idx in {0, len(epochs) - 1}:
            axes[1].annotate(str(int(epoch)), (epoch, samples_per_sec[idx]), xytext=(4, 4), textcoords="offset points", fontsize=7)

    fig.tight_layout()
    out_png = Path(args.output_png).resolve()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png)
    plt.close(fig)
    print(out_png)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
