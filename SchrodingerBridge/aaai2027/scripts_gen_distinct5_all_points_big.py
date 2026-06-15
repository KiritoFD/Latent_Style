from __future__ import annotations

import csv
import math
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parent
REPO_ROOT = ROOT.parents[1]
DOC_ROOT = REPO_ROOT / "SchrodingerBridge" / "docs" / "experiments"
BEST_CSV = REPO_ROOT / "best.csv"

LEGACY_TRANSFER_CSV = (
    DOC_ROOT
    / "distinct5_512_20260602"
    / "tables"
    / "clip_style_vs_1lpips_full_transfer_points.csv"
)
INMORTAL_EPOCH_CSV = DOC_ROOT / "2026-06-07-inmortal-epoch-eval-table.csv"
RESULTS_MASTER_CSV = DOC_ROOT / "aaai2027_results_master.csv"
PHASE2_POINTS_CSV = DOC_ROOT / "phase2_fiber_bundle" / "plot_points.csv"

OUT_CSV = ROOT / "fig_distinct5_all_points_big.csv"
OUT_PNG = ROOT / "fig_distinct5_all_points_big.png"
OUT_PDF = ROOT / "fig_distinct5_all_points_big.pdf"


plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif"],
        "font.size": 10.0,
        "axes.labelsize": 12.0,
        "axes.titlesize": 15.0,
        "xtick.labelsize": 10.5,
        "ytick.labelsize": 10.5,
        "legend.fontsize": 9.6,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.22,
        "grid.linewidth": 0.6,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.06,
    }
)


GROUP_STYLE = {
    "Reference": {"color": "#8E63C0", "marker": "o", "size": 34, "alpha": 1.0, "zorder": 8},
    "LBM legacy": {"color": "#D64045", "marker": "D", "size": 54, "alpha": 0.95, "zorder": 6},
    "LBM misc": {"color": "#4B5563", "marker": "X", "size": 70, "alpha": 0.95, "zorder": 7},
    "SaMAM": {"color": "#2F7DB7", "marker": "o", "size": 48, "alpha": 0.90, "zorder": 5},
    "SaMST": {"color": "#2CA02C", "marker": "s", "size": 54, "alpha": 0.92, "zorder": 6},
    "SaMAM-latent": {"color": "#0F766E", "marker": "o", "size": 46, "alpha": 0.92, "zorder": 6},
    "SaMST-latent": {"color": "#A855F7", "marker": "P", "size": 56, "alpha": 0.92, "zorder": 6},
    "Kinetic-only": {"color": "#B7791F", "marker": "o", "size": 26, "alpha": 0.65, "zorder": 3},
    "XPred core": {"color": "#2563EB", "marker": "o", "size": 26, "alpha": 0.60, "zorder": 3},
    "XPred proximal": {"color": "#7C3AED", "marker": "o", "size": 26, "alpha": 0.60, "zorder": 3},
    "XPred pattn/stokes": {"color": "#C2410C", "marker": "o", "size": 28, "alpha": 0.65, "zorder": 4},
    "Fiber Bundle": {"color": "#E08E00", "marker": "P", "size": 66, "alpha": 0.95, "zorder": 8},
}


def _safe_float(value: object) -> float | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        number = float(text)
    except ValueError:
        return None
    if not math.isfinite(number):
        return None
    return number


def _superfamily(family: str, variant: str = "") -> str:
    if family == "Reference":
        return "Reference"
    if family == "LANCET":
        return "LBM legacy"
    if family == "SaMAM":
        return "SaMAM"
    if family == "SaMST":
        return "SaMST"
    if family == "SaMAM-latent":
        return "SaMAM-latent"
    if family == "SaMST-latent":
        return "SaMST-latent"
    if family.startswith("K_"):
        return "Kinetic-only"
    if family in {"XPred_Barycenter", "XPred_Kmanifold"}:
        return "XPred core"
    if "Pattn" in family or "Stokes" in family:
        return "XPred pattn/stokes"
    if family.startswith("XPred_"):
        return "XPred proximal"
    if family == "LBM":
        return "LBM misc"
    if family in {"FiberBundle", "Phase2", "LBM-Phase2"}:
        return "Fiber Bundle"
    if variant.startswith("XPred_"):
        if "Pattn" in variant or "Stokes" in variant:
            return "XPred pattn/stokes"
        if variant in {"XPred_Barycenter", "XPred_Kmanifold"}:
            return "XPred core"
        return "XPred proximal"
    return "LBM misc"


def _add_row(rows: list[dict[str, object]], **kwargs: object) -> None:
    clip_style = _safe_float(kwargs.get("clip_style"))
    content_lpips = _safe_float(kwargs.get("content_lpips"))
    if clip_style is None or content_lpips is None:
        return
    family = str(kwargs.get("family") or "")
    variant = str(kwargs.get("variant") or "")
    row = dict(kwargs)
    row["clip_style"] = clip_style
    row["content_lpips"] = content_lpips
    row["one_minus_lpips"] = 1.0 - content_lpips
    row["train_min"] = _safe_float(kwargs.get("train_min"))
    row["train_time_sec"] = _safe_float(kwargs.get("train_time_sec"))
    row["order"] = _safe_float(kwargs.get("order"))
    row["superfamily"] = _superfamily(family, variant)
    rows.append(row)


def load_legacy_transfer(rows: list[dict[str, object]]) -> None:
    with LEGACY_TRANSFER_CSV.open("r", encoding="utf-8", newline="") as f:
        for raw in csv.DictReader(f):
            if raw["scope"] != "transfer":
                continue
            family = raw["family"]
            label = raw["label"]
            if family == "SaMAM" and _safe_float(raw["step_or_epoch"]) is not None and float(raw["step_or_epoch"]) > 2250:
                # The paper-safe Distinct5 SaMAM packet currently closes at 2250.
                continue
            trace_id = family
            if family == "LANCET":
                trace_id = ""
            _add_row(
                rows,
                point_id=f"legacy::{family}::{label}",
                source="legacy_transfer",
                family=family,
                variant=label,
                label=label,
                trace_id=trace_id,
                clip_style=raw["clip_style"],
                content_lpips=raw["content_lpips"],
                train_min=raw["train_min"],
                train_time_sec=_safe_float(raw["train_min"]) * 60.0 if _safe_float(raw["train_min"]) is not None else None,
                order=raw["step_or_epoch"],
                note=raw.get("note", ""),
            )


def load_inmortal_epochs(rows: list[dict[str, object]]) -> None:
    with INMORTAL_EPOCH_CSV.open("r", encoding="utf-8", newline="") as f:
        for raw in csv.DictReader(f):
            if not raw.get("clip_style") or not raw.get("content_lpips"):
                continue
            family = raw["family"]
            run_name = raw["run_name"]
            epoch = raw["epoch"]
            epoch_num = _safe_float(epoch.replace("epoch_", "")) if epoch.startswith("epoch_") else _safe_float(epoch)
            _add_row(
                rows,
                point_id=f"inmortal::{run_name}::{epoch}",
                source="inmortal_epoch",
                family=family,
                variant=run_name,
                label=f"{family} {epoch}",
                trace_id=run_name,
                clip_style=raw["clip_style"],
                content_lpips=raw["content_lpips"],
                train_min=_safe_float(raw["train_time_sec"]) / 60.0 if _safe_float(raw["train_time_sec"]) is not None else None,
                train_time_sec=raw["train_time_sec"],
                order=epoch_num,
                note=raw.get("summary_path", ""),
            )


def load_results_master_extras(rows: list[dict[str, object]]) -> None:
    duplicate_variants = {"F_e1", "H_e1", "H_e2", "K_e1"}
    seen_ids = {str(row.get("point_id")) for row in rows}
    with RESULTS_MASTER_CSV.open("r", encoding="utf-8", newline="") as f:
        for raw in csv.DictReader(f):
            if raw.get("dataset") != "distinct5_512" or raw.get("metric_surface") != "transfer":
                continue
            experiment = raw["experiment"]
            method = raw["method"]
            variant = raw["variant"]
            if experiment in seen_ids:
                continue
            if experiment.startswith("inmortal_"):
                continue
            if method in {"SaMAM", "SaMST"}:
                continue
            if method == "LBM" and variant in duplicate_variants:
                continue
            if method == "SaMAM-latent" and variant == "convergence":
                continue
            if method == "SaMST-latent" and variant == "convergence":
                continue
            family = method
            label = variant
            trace_id = ""
            if method == "LBM":
                family = "LBM"
                label = variant
            _add_row(
                rows,
                point_id=experiment,
                source="results_master_extra",
                family=family,
                variant=variant,
                label=label,
                trace_id=trace_id,
                clip_style=raw["clip_style"],
                content_lpips=raw["content_lpips"],
                train_min=raw["train_wall"],
                train_time_sec=_safe_float(raw["train_wall"]) * 60.0 if _safe_float(raw["train_wall"]) is not None else None,
                order=None,
                note=raw.get("selection", ""),
            )


def load_latent_manual_curves(rows: list[dict[str, object]]) -> None:
    samam_points = [
        ("20", 20.0, 1.89, 0.6297173805038135, 0.7823172304333333),
        ("110", 110.0, 10.41, 0.6388333174089590, 0.7041577109166667),
        ("300", 300.0, 27.75, 0.6222609871625899, 0.5650466211666666),
        ("600", 600.0, 56.97, 0.6540945671995480, 0.5467907414833334),
        ("1000", 1000.0, 96.99, 0.6667274163166682, 0.27443615400166665),
        ("1200", 1200.0, 114.41, 0.6549566574891408, 0.17385349117999999),
        ("1300", 1300.0, 123.01, 0.6532902393241724, 0.21977372080833332),
        ("1500", 1500.0, 140.65, 0.6547481072942416, 0.163526222025),
    ]
    for step_label, order, train_min, clip_style, content_lpips in samam_points:
        _add_row(
            rows,
            point_id=f"latent_curve::SaMAM::{step_label}",
            source="latent_manual_curve",
            family="SaMAM-latent",
            variant="curve",
            label=f"Lat SaMAM {step_label}",
            trace_id="SaMAM-latent-curve",
            clip_style=clip_style,
            content_lpips=content_lpips,
            train_min=train_min,
            train_time_sec=train_min * 60.0,
            order=order,
            note="manual curve from retained latent convergence plot",
        )

    samst_points = [
        ("b50", 50.0, 1.7763, 0.6104393125, 0.7295783353),
        ("b300", 300.0, 9.3050, 0.6104393125, 0.7295783353),
        ("950", 950.0, None, 0.6944, 0.8409),
        ("1050", 1050.0, None, 0.6819825260837873, 0.8318358248166667),
    ]
    for step_label, order, train_min, clip_style, content_lpips in samst_points:
        _add_row(
            rows,
            point_id=f"latent_curve::SaMST::{step_label}",
            source="latent_manual_curve",
            family="SaMST-latent",
            variant="curve",
            label=f"Lat SaMST {step_label}",
            trace_id="SaMST-latent-curve",
            clip_style=clip_style,
            content_lpips=content_lpips,
            train_min=train_min,
            train_time_sec=train_min * 60.0 if train_min is not None else None,
            order=order,
            note="manual curve from retained latent same-cost/convergence plots",
        )


def load_phase2_plot_points(rows: list[dict[str, object]]) -> None:
    if not PHASE2_POINTS_CSV.exists():
        return
    with PHASE2_POINTS_CSV.open("r", encoding="utf-8", newline="") as f:
        for raw in csv.DictReader(f):
            if raw.get("scope") != "transfer":
                continue
            point_id = raw.get("point_id") or f"phase2::{raw.get('variant', '')}::{raw.get('step_or_epoch', '')}"
            _add_row(
                rows,
                point_id=point_id,
                source="phase2_fiber_bundle_plot",
                family=raw.get("family", "FiberBundle"),
                variant=raw.get("variant", ""),
                label=raw.get("label", ""),
                trace_id=raw.get("trace_id", ""),
                clip_style=raw.get("clip_style"),
                content_lpips=raw.get("content_lpips"),
                train_min=raw.get("train_min"),
                train_time_sec=raw.get("train_time_sec"),
                order=raw.get("step_or_epoch"),
                note=raw.get("note", ""),
                label_dx=raw.get("label_dx", ""),
                label_dy=raw.get("label_dy", ""),
                source_summary=raw.get("source_summary", ""),
                style_minus_idt=raw.get("style_minus_idt", ""),
            )


def collect_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    load_legacy_transfer(rows)
    load_inmortal_epochs(rows)
    load_results_master_extras(rows)
    load_latent_manual_curves(rows)
    load_phase2_plot_points(rows)
    return rows


def load_best_rows() -> list[dict[str, str]]:
    if not BEST_CSV.exists():
        return []
    with BEST_CSV.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def best_csv_point_id(best_row: dict[str, str]) -> str | None:
    source_table = str(best_row.get("source_table", ""))
    experiment = str(best_row.get("experiment", ""))
    selection = str(best_row.get("selection", ""))
    if not experiment:
        return None
    if source_table.endswith("2026-06-07-inmortal-epoch-eval-table.csv"):
        if selection:
            return f"inmortal::{experiment}::{selection}"
        return None
    if source_table.endswith("aaai2027_results_master.csv"):
        return experiment
    return None


def pareto_frontier(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    frontier_rows = [row for row in rows if row["family"] != "Reference"]
    ordered = sorted(frontier_rows, key=lambda row: (float(row["one_minus_lpips"]), float(row["clip_style"])))
    frontier: list[dict[str, object]] = []
    best_style = -1.0
    for row in reversed(ordered):
        style = float(row["clip_style"])
        if style > best_style + 1e-12:
            frontier.append(row)
            best_style = style
    frontier.reverse()
    return frontier


def write_unified_csv(rows: list[dict[str, object]]) -> None:
    fields = [
        "point_id",
        "source",
        "superfamily",
        "family",
        "variant",
        "label",
        "trace_id",
        "clip_style",
        "content_lpips",
        "one_minus_lpips",
        "train_min",
        "train_time_sec",
        "order",
        "note",
        "label_dx",
        "label_dy",
        "source_summary",
        "style_minus_idt",
    ]
    with OUT_CSV.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def plot(rows: list[dict[str, object]]) -> None:
    fig, ax = plt.subplots(figsize=(14.4, 9.6))
    ax.set_facecolor("#FBFAF7")

    by_trace: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        trace_id = str(row.get("trace_id") or "")
        if trace_id:
            by_trace[trace_id].append(row)

    for trace_rows in by_trace.values():
        if len(trace_rows) < 2:
            continue
        trace_rows = sorted(
            trace_rows,
            key=lambda row: (
                float(row["order"]) if row.get("order") is not None else 1e9,
                float(row["train_min"]) if row.get("train_min") is not None else 1e9,
            ),
        )
        superfamily = str(trace_rows[0]["superfamily"])
        style = GROUP_STYLE[superfamily]
        ax.plot(
            [float(row["one_minus_lpips"]) for row in trace_rows],
            [float(row["clip_style"]) for row in trace_rows],
            color=style["color"],
            lw=1.2 if superfamily not in {"XPred pattn/stokes", "XPred core"} else 1.5,
            alpha=0.28 if superfamily.startswith("XPred") or superfamily == "Kinetic-only" else 0.60,
            zorder=2,
        )

    legend_handles = []
    legend_labels = []
    for superfamily, style in GROUP_STYLE.items():
        group_rows = [row for row in rows if row["superfamily"] == superfamily]
        if not group_rows:
            continue
        scatter = ax.scatter(
            [float(row["one_minus_lpips"]) for row in group_rows],
            [float(row["clip_style"]) for row in group_rows],
            s=style["size"],
            c=style["color"],
            marker=style["marker"],
            alpha=style["alpha"],
            edgecolors="white" if superfamily != "Reference" else style["color"],
            linewidths=0.5 if superfamily != "Reference" else 1.6,
            zorder=style["zorder"],
            label=superfamily,
        )
        legend_handles.append(scatter)
        legend_labels.append(f"{superfamily} ({len(group_rows)})")

    frontier = pareto_frontier(rows)
    ax.plot(
        [float(row["one_minus_lpips"]) for row in frontier],
        [float(row["clip_style"]) for row in frontier],
        color="black",
        lw=1.8,
        alpha=0.78,
        zorder=9,
    )
    ax.scatter(
        [float(row["one_minus_lpips"]) for row in frontier],
        [float(row["clip_style"]) for row in frontier],
        s=28,
        c="black",
        zorder=10,
    )

    idt = next(row for row in rows if row["family"] == "Reference")
    idt_style = float(idt["clip_style"])
    ax.axhline(idt_style, color="#8E63C0", lw=1.25, ls=(0, (7, 4)), alpha=0.88, zorder=1)
    ax.text(0.992, idt_style + 0.0035, "IDT transfer floor", fontsize=10.2, color="#6F46A5", ha="right")

    highlight_ids = {
        "legacy::LANCET::F e1": ("LBM-F", 8, -14, "#D64045"),
        "legacy::LANCET::H e2": ("LBM-H", -12, 8, "#D64045"),
        "legacy::LANCET::K e1": ("LBM-K", 8, 10, "#D64045"),
        "legacy::SaMAM::SaMAM 2250": ("SaMAM 2250", 8, 8, "#2F7DB7"),
        "legacy::SaMST::SaMST e15": ("SaMST e15", 10, -12, "#2CA02C"),
        "latent_curve::SaMAM::1500": ("Lat SaMAM 1500", 10, -12, "#0F766E"),
        "latent_curve::SaMST::1050": ("Lat SaMST 1050", 10, -12, "#A855F7"),
    }
    best_slot_labels = {
        "style_best_current": ("style best", 8, -16, "#1D4ED8"),
        "best_promoted_style_current": ("promoted style", 10, 10, "#0F52BA"),
        "balanced_best_style_ge_072": ("best >=0.72", 10, 8, "#9A3412"),
        "best_promoted_balanced_current": ("promoted balance", 10, -16, "#7C2D12"),
        "best_lpips_style_ge_070": ("best LPIPS >=0.70", 10, -14, "#B45309"),
        "best_promoted_lpips_ge_070": ("promoted LPIPS", 10, 8, "#92400E"),
        "best_structot_tradeoff": ("StructOT", 8, 10, "#92400E"),
        "best_kinetic_only_control": ("kinetic best", 10, 8, "#A16207"),
        "best_compact_style_anchor": ("compact K", 8, 10, "#D64045"),
        "best_compact_lpips_anchor": ("compact F", 8, -14, "#D64045"),
        "best_compact_mainline_anchor": ("compact K", 8, 10, "#D64045"),
    }
    by_id = {str(row["point_id"]): row for row in rows}
    for best_row in load_best_rows():
        point_id = best_csv_point_id(best_row)
        if point_id is None:
            continue
        slot = str(best_row.get("slot", ""))
        label = best_slot_labels.get(slot)
        if label is None:
            continue
        highlight_ids[point_id] = label
    for point_id, (text, dx, dy, color) in highlight_ids.items():
        row = by_id.get(point_id)
        if row is None:
            continue
        ax.annotate(
            text,
            (float(row["one_minus_lpips"]), float(row["clip_style"])),
            xytext=(dx, dy),
            textcoords="offset points",
            fontsize=9.2,
            color=color,
            bbox=dict(boxstyle="round,pad=0.18", fc="white", ec=color, lw=0.6, alpha=0.92),
            arrowprops=dict(arrowstyle="-", color=color, lw=0.6, shrinkA=2, shrinkB=2),
            zorder=11,
        )
    for row in rows:
        if row.get("source") != "phase2_fiber_bundle_plot":
            continue
        text = str(row.get("label") or "").strip()
        if not text:
            continue
        dx = _safe_float(row.get("label_dx"))
        dy = _safe_float(row.get("label_dy"))
        ax.annotate(
            text,
            (float(row["one_minus_lpips"]), float(row["clip_style"])),
            xytext=(dx if dx is not None else 8.0, dy if dy is not None else 10.0),
            textcoords="offset points",
            fontsize=9.0,
            color="#9A5B00",
            bbox=dict(boxstyle="round,pad=0.16", fc="white", ec="#E08E00", lw=0.6, alpha=0.90),
            arrowprops=dict(arrowstyle="-", color="#E08E00", lw=0.55, shrinkA=2, shrinkB=2),
            zorder=12,
        )

    x_vals = [float(row["one_minus_lpips"]) for row in rows]
    y_vals = [float(row["clip_style"]) for row in rows]
    ax.set_xlim(max(0.12, min(x_vals) - 0.03), min(1.02, max(x_vals) + 0.02))
    ax.set_ylim(min(y_vals) - 0.025, max(y_vals) + 0.025)
    ax.set_xlabel("1 - LPIPS")
    ax.set_ylabel("Transfer CLIP-style")
    ax.set_title("Distinct5-512 transfer landscape with all recorded operating points", fontweight="bold")
    ax.text(
        0.015,
        0.985,
        f"{len(rows)} points | legacy + latent + inmortal + paper-facing extras",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=10.4,
        color="#333333",
        bbox=dict(boxstyle="round,pad=0.22", fc="white", ec="#BBBBBB", lw=0.6, alpha=0.92),
    )
    ax.text(
        0.985,
        0.02,
        "upper-right is better",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=10.0,
        color="#333333",
        bbox=dict(boxstyle="round,pad=0.18", fc="white", ec="#BBBBBB", lw=0.6, alpha=0.92),
    )
    ax.legend(
        legend_handles,
        legend_labels,
        loc="lower right",
        bbox_to_anchor=(0.985, 0.02),
        borderaxespad=0.0,
        frameon=False,
    )

    fig.savefig(OUT_PDF)
    fig.savefig(OUT_PNG)
    plt.close(fig)


def main() -> None:
    rows = collect_rows()
    rows = sorted(
        rows,
        key=lambda row: (
            str(row["superfamily"]),
            str(row["trace_id"]),
            float(row["order"]) if row.get("order") is not None else 1e9,
            float(row["train_min"]) if row.get("train_min") is not None else 1e9,
            str(row["label"]),
        ),
    )
    write_unified_csv(rows)
    plot(rows)
    print(f"rows={len(rows)}")
    print(OUT_CSV)
    print(OUT_PNG)
    print(OUT_PDF)


if __name__ == "__main__":
    main()
