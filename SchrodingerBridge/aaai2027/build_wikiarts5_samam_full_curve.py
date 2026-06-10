from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patheffects as pe


ROOT = Path(__file__).resolve().parent
WORKSPACE = ROOT.parent.parent
RESULT_ROOT = WORKSPACE / "Related_Works" / "baseline_pipeline" / "results" / "samam_wikiarts5_patch8_segmented_20260610_094447"
CURVE_CSV = RESULT_ROOT / "curve_metrics.csv"
CONV_JSON = RESULT_ROOT / "curve_convergence.json"
OUT_DIR = ROOT / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_PNG = OUT_DIR / "fig_wikiarts5_samam_full_curve.png"
OUT_PDF = OUT_DIR / "fig_wikiarts5_samam_full_curve.pdf"
OUT_JSON = ROOT / "samam_wikiarts5_full_curve_summary.json"

IDT_TRANSFER = 0.6399224616587162


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _annotate(ax, x: float, y: float, text: str, dx: float, dy: float, color: str) -> None:
    ax.annotate(
        text,
        (x, y),
        xytext=(dx, dy),
        textcoords="offset points",
        fontsize=7.0,
        fontweight="semibold",
        color=color,
        arrowprops=dict(arrowstyle="-", color=color, lw=0.6, shrinkA=2, shrinkB=3),
        path_effects=[pe.withStroke(linewidth=2.2, foreground="white")],
    )


def main() -> int:
    rows = _read_csv(CURVE_CSV)
    conv = json.loads(CONV_JSON.read_text(encoding="utf-8"))

    steps = [int(float(row["step"])) for row in rows]
    tr_style = [float(row["transfer_clip_style"]) for row in rows]
    ap_style = [float(row["all_pairs_clip_style"]) for row in rows]
    tr_lpips = [float(row["transfer_lpips"]) for row in rows]
    ap_lpips = [float(row["all_pairs_lpips"]) for row in rows]

    best_clip_step = int(conv["best_step"])
    best_lpips_step = max(rows, key=lambda r: (-float(r["transfer_lpips"]), float(r["transfer_clip_style"])))["step"]
    best_lpips_step = int(float(best_lpips_step))
    last_pareto_step = int(conv["last_pareto_step"])
    latest_step = int(conv["newest_step"])

    step_to_idx = {step: idx for idx, step in enumerate(steps)}

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "DejaVu Serif"],
            "font.size": 9.0,
            "axes.labelsize": 9.2,
            "axes.titlesize": 10.0,
            "xtick.labelsize": 7.7,
            "ytick.labelsize": 7.7,
            "legend.fontsize": 7.2,
            "figure.dpi": 300,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.03,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.24,
            "grid.linewidth": 0.55,
        }
    )

    fig, axes = plt.subplots(2, 1, figsize=(8.2, 6.1), dpi=180, sharex=True)

    # Style panel
    ax = axes[0]
    ax.set_facecolor("#FCFBF8")
    ax.plot(steps, tr_style, color="#D64045", linewidth=1.7, marker="o", markersize=3.2, label="transfer CLIP-S")
    ax.plot(steps, ap_style, color="#1D4ED8", linewidth=1.5, marker="s", markersize=3.0, alpha=0.92, label="all-pairs CLIP-S")
    ax.axhline(IDT_TRANSFER, color="#8E63C0", linewidth=1.3, linestyle=(0, (7, 4)), label="IDT transfer CLIP-S")
    ax.scatter([best_clip_step], [tr_style[step_to_idx[best_clip_step]]], s=80, c="#D64045", marker="D", edgecolors="white", linewidths=0.9, zorder=5)
    ax.scatter([latest_step], [tr_style[step_to_idx[latest_step]]], s=74, c="#16A085", marker="^", edgecolors="white", linewidths=0.9, zorder=5)
    ax.axvline(last_pareto_step, color="#B45309", linewidth=1.0, linestyle="--", alpha=0.9)
    _annotate(ax, best_clip_step, tr_style[step_to_idx[best_clip_step]], "best transfer style", 8, 10, "#D64045")
    _annotate(ax, latest_step, tr_style[step_to_idx[latest_step]], "latest", -38, -18, "#16A085")
    _annotate(ax, last_pareto_step, tr_style[step_to_idx[last_pareto_step]], "last Pareto", -62, 12, "#B45309")
    ax.set_ylabel("CLIP-S")
    ax.set_title("SaMAM wikiarts5 full convergence curve")
    ax.legend(loc="lower right")

    # LPIPS panel
    ax = axes[1]
    ax.set_facecolor("#FBFBFB")
    ax.plot(steps, tr_lpips, color="#D64045", linewidth=1.7, marker="o", markersize=3.2, label="transfer LPIPS")
    ax.plot(steps, ap_lpips, color="#1D4ED8", linewidth=1.5, marker="s", markersize=3.0, alpha=0.92, label="all-pairs LPIPS")
    ax.scatter([best_lpips_step], [tr_lpips[step_to_idx[best_lpips_step]]], s=80, c="#B45309", marker="D", edgecolors="white", linewidths=0.9, zorder=5)
    ax.scatter([latest_step], [tr_lpips[step_to_idx[latest_step]]], s=74, c="#16A085", marker="^", edgecolors="white", linewidths=0.9, zorder=5)
    ax.axvline(last_pareto_step, color="#B45309", linewidth=1.0, linestyle="--", alpha=0.9)
    _annotate(ax, best_lpips_step, tr_lpips[step_to_idx[best_lpips_step]], "best transfer LPIPS", -116, -6, "#B45309")
    _annotate(ax, latest_step, tr_lpips[step_to_idx[latest_step]], "latest", -38, 8, "#16A085")
    ax.set_xlabel("training step")
    ax.set_ylabel("LPIPS")
    ax.legend(loc="upper right")

    status_text = (
        f"best_transfer_style_step={best_clip_step}, "
        f"best_transfer_lpips_step={best_lpips_step}, "
        f"last_pareto_step={last_pareto_step}, "
        f"latest_step={latest_step}, "
        f"since_last_pareto={conv['since_last_pareto']}, "
        f"tail_flat={conv['tail_flat']}, "
        f"converged={conv['converged']}"
    )
    fig.text(0.5, 0.01, status_text, ha="center", va="bottom", fontsize=7.2, color="#444444")

    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(OUT_PNG)
    fig.savefig(OUT_PDF)
    plt.close(fig)

    summary = {
        "curve_csv": str(CURVE_CSV),
        "convergence_json": str(CONV_JSON),
        "best_transfer_style_step": best_clip_step,
        "best_transfer_lpips_step": best_lpips_step,
        "last_pareto_step": last_pareto_step,
        "latest_step": latest_step,
        "since_last_pareto": int(conv["since_last_pareto"]),
        "tail_flat": bool(conv["tail_flat"]),
        "converged": bool(conv["converged"]),
        "output_png": str(OUT_PNG),
        "output_pdf": str(OUT_PDF),
    }
    OUT_JSON.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(OUT_PNG)
    print(OUT_JSON)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
