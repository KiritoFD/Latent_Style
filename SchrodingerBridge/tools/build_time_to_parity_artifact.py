"""Build the Distinct5 normalized time-to-parity artifact.

This script materializes the reviewer-safe timing packet requested by
`docs/experiments/2026-06-03-time-to-parity/README.md`:

- a canonical CSV with explicit timing semantics
- vector figures for clip_style, content_lpips, and delta_idt

The artifact intentionally uses one explicit scope only:

- dataset: Distinct5-512
- evaluation scope: full 5x5 / 750 outputs

LBM is represented by reviewed operating-point records.
SaMAM is represented by the currently indexed Distinct5 partial curve.
SaMST is represented by the currently indexed Distinct5 operating point.
"""

from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
DOC_ROOT = ROOT / "docs" / "experiments"
FULL_TRANSFER_CSV = DOC_ROOT / "distinct5_512_20260602" / "tables" / "clip_style_vs_1lpips_full_transfer_points.csv"
ARTFID_POINTS_CSV = DOC_ROOT / "comparison_20260602" / "artfid_comparison_points.csv"
OUT_DIR = DOC_ROOT / "2026-06-03-time-to-parity"
OUT_CSV = OUT_DIR / "distinct5_time_to_parity_points.csv"
FIG_DIR = OUT_DIR / "figures"


plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif"],
        "font.size": 8.8,
        "axes.labelsize": 9.2,
        "axes.titlesize": 9.4,
        "xtick.labelsize": 8.0,
        "ytick.labelsize": 8.0,
        "legend.fontsize": 7.8,
        "legend.frameon": False,
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.04,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.18,
        "grid.linewidth": 0.55,
        "lines.linewidth": 1.75,
        "lines.markersize": 4.8,
    }
)


COLORS = {
    "lbm": "#E76F51",
    "samam": "#264653",
    "samst": "#7A4FA2",
    "ref": "#8C8C8C",
    "zero": "#A0A0A0",
}


LBM_VARIANTS = {
    "F e1": ("F_best_lpips", "Current best LPIPS operating point"),
    "H e1": ("H_balanced_lpips", "Balanced LPIPS operating point"),
    "H e2": ("H_balanced_style", "Balanced style operating point"),
    "K e1": ("K_best_style", "Current best style operating point"),
}


FIELDS = [
    "date",
    "method",
    "variant",
    "dataset",
    "scope",
    "checkpoint_or_step",
    "wall_seconds",
    "timing_mode",
    "includes_eval",
    "eval_scope",
    "eval_wall_seconds",
    "clip_style",
    "content_lpips",
    "delta_idt_full",
    "delta_idt_transfer",
    "hardware",
    "status",
    "timing_quality_flag",
    "evidence_path",
    "note",
]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _to_float(value: str | None) -> float | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def _format_float(value: float | None, digits: int = 10) -> str:
    if value is None or not math.isfinite(value):
        return ""
    return f"{value:.{digits}f}".rstrip("0").rstrip(".")


def _minutes_label(minutes: float) -> str:
    if minutes >= 60.0:
        return f"{minutes / 60.0:.1f}h"
    return f"{minutes:.1f}m"


def _annotate(ax, x: float, y: float, text: str, dx: float, dy: float) -> None:
    ax.annotate(
        text,
        (x, y),
        xytext=(dx, dy),
        textcoords="offset points",
        ha="left" if dx >= 0 else "right",
        va="center",
        fontsize=6.9,
        color="#333333",
        arrowprops=dict(arrowstyle="-", color="#777777", lw=0.45, shrinkA=0, shrinkB=3),
    )


def _set_time_axis(ax) -> None:
    ticks = [1, 2, 5, 10, 30, 60, 120, 240, 480]
    labels = ["1", "2", "5", "10", "30", "1h", "2h", "4h", "8h"]
    ax.set_xscale("log")
    ax.set_xlim(0.9, 600.0)
    ax.set_xticks(ticks)
    ax.set_xticklabels(labels)


def collect_rows() -> list[dict[str, str]]:
    full_transfer = read_csv(FULL_TRANSFER_CSV)
    artfid_rows = read_csv(ARTFID_POINTS_CSV)

    by_scope_family_label = {
        (row["scope"], row["family"], row["label"]): row for row in full_transfer
    }
    transfer_lookup = {
        (row["family"], row["label"]): row
        for row in full_transfer
        if row["scope"] == "transfer"
    }

    idt_full = by_scope_family_label[("full", "Reference", "No-op 5x5")]
    idt_transfer = by_scope_family_label[("transfer", "Reference", "No-op transfer")]
    idt_full_style = float(idt_full["clip_style"])
    idt_transfer_style = float(idt_transfer["clip_style"])

    rows: list[dict[str, str]] = []

    rows.append(
        {
            "date": "2026-06-03",
            "method": "idt",
            "variant": "no_op_reference",
            "dataset": "distinct5_512",
            "scope": "full",
            "checkpoint_or_step": "unchanged",
            "wall_seconds": "0",
            "timing_mode": "reference",
            "includes_eval": "false",
            "eval_scope": "full 5x5 / 750",
            "eval_wall_seconds": "",
            "clip_style": _format_float(idt_full_style),
            "content_lpips": _format_float(float(idt_full["content_lpips"])),
            "delta_idt_full": _format_float(0.0),
            "delta_idt_transfer": _format_float(0.0),
            "hardware": "reference / eval bundle",
            "status": "completed",
            "timing_quality_flag": "reference_only",
            "evidence_path": str(FULL_TRANSFER_CSV),
            "note": "Unchanged-image reference; timing is defined as zero and excluded from parity claims.",
        }
    )

    for label, (variant, note) in LBM_VARIANTS.items():
        full_row = by_scope_family_label[("full", "LANCET", label)]
        transfer_row = transfer_lookup.get(("LANCET", label))
        full_style = float(full_row["clip_style"])
        transfer_style = float(transfer_row["clip_style"]) if transfer_row else None
        rows.append(
            {
                "date": "2026-06-03",
                "method": "LBM",
                "variant": variant,
                "dataset": "distinct5_512",
                "scope": "full",
                "checkpoint_or_step": full_row["step_or_epoch"],
                "wall_seconds": _format_float(float(full_row["train_min"]) * 60.0, digits=6),
                "timing_mode": "operating_point_record",
                "includes_eval": "false",
                "eval_scope": "full 5x5 / 750",
                "eval_wall_seconds": "",
                "clip_style": _format_float(full_style),
                "content_lpips": _format_float(float(full_row["content_lpips"])),
                "delta_idt_full": _format_float(full_style - idt_full_style),
                "delta_idt_transfer": _format_float(transfer_style - idt_transfer_style if transfer_style is not None else None),
                "hardware": "remote RTX 3060",
                "status": "completed",
                "timing_quality_flag": "normal",
                "evidence_path": str(FULL_TRANSFER_CSV),
                "note": note + "; cumulative training wall excludes eval.",
            }
        )

    samam_full_rows = [
        row for row in full_transfer if row["scope"] == "full" and row["family"] == "SaMAM"
    ]
    for row in samam_full_rows:
        transfer_row = transfer_lookup.get(("SaMAM", row["label"]))
        full_style = float(row["clip_style"])
        transfer_style = float(transfer_row["clip_style"]) if transfer_row else None
        rows.append(
            {
                "date": "2026-06-03",
                "method": "SaMAM",
                "variant": "distinct5_partial_curve",
                "dataset": "distinct5_512",
                "scope": "full",
                "checkpoint_or_step": row["step_or_epoch"],
                "wall_seconds": _format_float(float(row["train_min"]) * 60.0, digits=6),
                "timing_mode": "full_curve_partial",
                "includes_eval": "false",
                "eval_scope": "full 5x5 / 750",
                "eval_wall_seconds": "",
                "clip_style": _format_float(full_style),
                "content_lpips": _format_float(float(row["content_lpips"])),
                "delta_idt_full": _format_float(full_style - idt_full_style),
                "delta_idt_transfer": _format_float(transfer_style - idt_transfer_style if transfer_style is not None else None),
                "hardware": "remote RTX 3060 WSL",
                "status": "completed",
                "timing_quality_flag": "normal",
                "evidence_path": str(FULL_TRANSFER_CSV),
                "note": f"{row['label']} partial-curve point; cumulative training wall excludes eval.",
            }
        )

    samst_full = next(
        row
        for row in artfid_rows
        if row["dataset"] == "distinct5_512" and row["scope"] == "full" and row["method"] == "SaMST"
    )
    samst_transfer = next(
        row
        for row in artfid_rows
        if row["dataset"] == "distinct5_512" and row["scope"] == "transfer" and row["method"] == "SaMST"
    )
    samst_wall_seconds = 5.8 * 3600.0
    samst_full_style = float(samst_full["clip_style"])
    samst_transfer_style = float(samst_transfer["clip_style"])
    rows.append(
        {
            "date": "2026-06-03",
            "method": "SaMST",
            "variant": "distinct5_e15",
            "dataset": "distinct5_512",
            "scope": "full",
            "checkpoint_or_step": "epoch_15",
            "wall_seconds": _format_float(samst_wall_seconds, digits=1),
            "timing_mode": "operating_point_record",
            "includes_eval": "false",
            "eval_scope": "full 5x5 / 750",
            "eval_wall_seconds": "",
            "clip_style": _format_float(samst_full_style),
            "content_lpips": _format_float(float(samst_full["content_lpips"])),
            "delta_idt_full": _format_float(samst_full_style - idt_full_style),
            "delta_idt_transfer": _format_float(samst_transfer_style - idt_transfer_style),
            "hardware": "Windows / reproduced run",
            "status": "completed",
            "timing_quality_flag": "operating_point_only",
            "evidence_path": samst_full["summary_path"],
            "note": "Single Distinct5 operating-point record; no same-scope curve yet, so this row is not a full trajectory.",
        }
    )

    def sort_key(row: dict[str, str]) -> tuple[int, float]:
        order = {"idt": 0, "LBM": 1, "SaMAM": 2, "SaMST": 3}
        return order.get(row["method"], 9), float(row["wall_seconds"])

    rows.sort(key=sort_key)
    return rows


def write_csv(rows: Iterable[dict[str, str]]) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with OUT_CSV.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def load_artifact_rows() -> list[dict[str, object]]:
    out: list[dict[str, object]] = []
    for row in read_csv(OUT_CSV):
        out.append(
            {
                **row,
                "wall_minutes": float(row["wall_seconds"]) / 60.0,
                "clip_style": float(row["clip_style"]),
                "content_lpips": float(row["content_lpips"]),
                "delta_idt_full": _to_float(row["delta_idt_full"]) or 0.0,
            }
        )
    return out


def plot_metric(rows: list[dict[str, object]], metric: str, ylabel: str, ref_y: float, out_name: str, ylimits: tuple[float, float]) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(4.4, 2.9))

    idt_rows = [r for r in rows if r["method"] == "idt"]
    lbm_rows = sorted([r for r in rows if r["method"] == "LBM"], key=lambda r: r["wall_minutes"])
    samam_rows = sorted([r for r in rows if r["method"] == "SaMAM"], key=lambda r: r["wall_minutes"])
    samst_rows = [r for r in rows if r["method"] == "SaMST"]

    if metric == "delta_idt_full":
        ax.axhline(0.0, color=COLORS["zero"], lw=1.15, ls="--", zorder=1, label="idt baseline")
    else:
        ax.axhline(ref_y, color=COLORS["ref"], lw=1.15, ls="--", zorder=1, label="idt reference")

    ax.plot(
        [r["wall_minutes"] for r in samam_rows],
        [r[metric] for r in samam_rows],
        color=COLORS["samam"],
        marker="o",
        label="SaMAM partial curve",
        zorder=2,
    )
    ax.scatter(
        [r["wall_minutes"] for r in lbm_rows],
        [r[metric] for r in lbm_rows],
        color=COLORS["lbm"],
        marker="D",
        edgecolor="white",
        linewidth=0.7,
        s=34,
        label="LBM operating points",
        zorder=3,
    )
    ax.scatter(
        [r["wall_minutes"] for r in samst_rows],
        [r[metric] for r in samst_rows],
        color=COLORS["samst"],
        marker="X",
        edgecolor="white",
        linewidth=0.65,
        s=48,
        label="SaMST operating point",
        zorder=4,
    )

    _set_time_axis(ax)
    ax.set_xlabel("Cumulative training wall time (min, log scale)")
    ax.set_ylabel(ylabel)
    ax.set_ylim(*ylimits)

    if metric != "delta_idt_full" and idt_rows:
        idt = idt_rows[0]
        label = "idt"
        y = ref_y
        ax.text(1.02, y + (0.005 if metric == "clip_style" else 0.012), label, color="#666666", fontsize=6.9)

    # Key annotations.
    by_variant = {str(r["variant"]): r for r in lbm_rows}
    lbm_annotations = [
        ("F_best_lpips", "LBM F e1\n1.2m", (8, 10)),
        ("H_balanced_style", "LBM H e2\n2.3m", (8, -10)),
    ]
    if metric == "clip_style":
        lbm_annotations.append(("K_best_style", "LBM K e1", (10, -20)))

    for variant, text, offset in lbm_annotations:
        row = by_variant.get(variant)
        if row:
            _annotate(ax, float(row["wall_minutes"]), float(row[metric]), text, *offset)

    if samam_rows:
        _annotate(
            ax,
            float(samam_rows[0]["wall_minutes"]),
            float(samam_rows[0][metric]),
            "SaMAM 250\n53.2m",
            8,
            -12,
        )
        _annotate(
            ax,
            float(samam_rows[-1]["wall_minutes"]),
            float(samam_rows[-1][metric]),
            "SaMAM 2250\n7.6h",
            8,
            10,
        )

    if samst_rows:
        samst = samst_rows[0]
        _annotate(
            ax,
            float(samst["wall_minutes"]),
            float(samst[metric]),
            "SaMST e15\n5.8h",
            -10,
            10 if metric != "content_lpips" else -12,
        )

    ax.legend(loc="best")
    fig.savefig(FIG_DIR / f"{out_name}.pdf")
    fig.savefig(FIG_DIR / f"{out_name}.png")
    plt.close(fig)


def main() -> None:
    rows = collect_rows()
    write_csv(rows)
    artifact_rows = load_artifact_rows()
    idt_clip = next(r["clip_style"] for r in artifact_rows if r["method"] == "idt")
    plot_metric(
        artifact_rows,
        metric="clip_style",
        ylabel="CLIP-style ↑",
        ref_y=float(idt_clip),
        out_name="distinct5_time_to_clip_style",
        ylimits=(0.53, 0.74),
    )
    plot_metric(
        artifact_rows,
        metric="content_lpips",
        ylabel="content LPIPS ↓",
        ref_y=0.0,
        out_name="distinct5_time_to_lpips",
        ylimits=(-0.02, 0.68),
    )
    plot_metric(
        artifact_rows,
        metric="delta_idt_full",
        ylabel=r"$\Delta$ CLIP-style vs idt ↑",
        ref_y=0.0,
        out_name="distinct5_time_to_delta_idt",
        ylimits=(-0.16, 0.06),
    )
    print(f"Wrote {OUT_CSV}")
    print(f"Wrote figures under {FIG_DIR}")


if __name__ == "__main__":
    main()
