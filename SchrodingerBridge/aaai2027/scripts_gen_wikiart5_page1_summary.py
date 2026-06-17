from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.patheffects as pe
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parent
WORKSPACE = ROOT.parent.parent
PHASE2_POINTS_CSV = WORKSPACE / "SchrodingerBridge" / "docs" / "experiments" / "phase2_fiber_bundle" / "plot_points.csv"
SAMAM_CURVE_CSV = WORKSPACE / "Related_Works" / "baseline_pipeline" / "results" / "samam_wikiarts5_patch8_segmented_20260610_094447" / "curve_metrics.csv"
SAMST_CURVE_CSV = WORKSPACE / "Related_Works" / "baseline_pipeline" / "results" / "samst_wikiarts5_wsl_20260610_172206" / "eval_bundle" / "clip_lpips_curve.csv"
POINTS_CSV = ROOT / "page1_bundle" / "wikiart5_page1_clip_lpips_points.csv"
OUT_DIR = ROOT / "figures"

DATASET_SURFACE = "wikiarts5_full_notest_train__distinct5_512_classview_test"
IDT_TRANSFER_CLIP = 0.6399208252628644


FIELDS = [
    "point_id",
    "dataset_surface",
    "scope",
    "source_group",
    "family",
    "variant",
    "label",
    "trace_id",
    "step_or_epoch",
    "clip_style",
    "content_lpips",
    "one_minus_lpips",
    "style_minus_idt",
    "train_min",
    "train_time_sec",
    "label_dx",
    "label_dy",
    "note",
    "source_summary",
]


TRACE_STYLES = {
    "idt": ("#8E63C0", "X", 82, 1.0),
    "seedream_test_only": ("#C58A2B", "P", 82, 1.0),
    "samam_wikiarts5_patch8": ("#5D8FBF", "o", 18, 0.66),
    "samst_wikiarts5": ("#55A85B", "s", 58, 0.9),
    "k070_e1_e5": ("#E08E00", "P", 56, 0.94),
    "pattn_enhanced_tok_e1_e10": ("#0F766E", "X", 52, 0.9),
    "fiber_sde_iso_scan": ("#2563EB", "o", 42, 0.78),
    "fiber_sde_fiber_scan": ("#DC2626", "D", 42, 0.78),
    "fiber_sde_fine_k070_e3": ("#B45309", "o", 48, 0.86),
    "rgbcal_k070_e3": ("#9CA3AF", "h", 42, 0.72),
    "topology_release_k070_e3": ("#7C3AED", "v", 42, 0.72),
    "appearance_blend_k070_e3": ("#9333EA", "v", 42, 0.72),
    "pc_lowpass_k070_e3": ("#0891B2", "d", 42, 0.82),
    "smoe_translator_k070_e3": ("#64748B", "^", 36, 0.72),
    "k070_kin070_vlen010": ("#EF4444", "x", 44, 0.78),
    "actuation_spatial_carriergate_k070_e3": ("#F97316", ">", 44, 0.82),
    "latent_affine_k070_e3": ("#BE123C", "^", 72, 0.98),
    "latent_affine_refine_k070_e3": ("#E11D48", "D", 48, 0.92),
    "latent_affine_pc_k070_e3": ("#DB2777", "v", 52, 0.90),
    "i2sb_pnp_fiber_sde_k070": ("#111827", "*", 78, 0.92),
    "i2sb_slerp_orthogonal_lowhigh_k070_e3": ("#EA580C", "h", 42, 0.78),
    "i2sb_orthogonal_lowanchor050_k070_e3": ("#F59E0B", "H", 48, 0.84),
    "i2sb_orthogonal_lowanchor065_k070_e3": ("#D97706", "p", 46, 0.84),
    "style_covariant_probe_parent_low050e9": ("#6B7280", "o", 34, 0.56),
    "style_covariant_probe_gaussian_low050e9": ("#7C3AED", "h", 40, 0.70),
    "style_covariant_probe_stylecov_low050e9": ("#0F766E", "D", 40, 0.76),
}

LEGEND_MAP = {
    "samam_wikiarts5_patch8": "SaMAM (WikiArt-5)",
    "samst_wikiarts5": "SaMST (WikiArt-5)",
    "seedream_test_only": "Seedream",
    "k070_e1_e5": "Ours k070",
    "fiber_sde_fine_k070_e3": "Fiber/SDE",
    "actuation_spatial_carriergate_k070_e3": "Carrier gate",
    "latent_affine_k070_e3": "Latent affine",
    "latent_affine_refine_k070_e3": "LatAff refine",
    "latent_affine_pc_k070_e3": "LatAff+PC",
    "i2sb_pnp_fiber_sde_k070": "I2SB combo",
    "i2sb_slerp_orthogonal_lowhigh_k070_e3": "I2SB slerp+orth",
    "i2sb_orthogonal_lowanchor050_k070_e3": "I2SB low-anchor",
    "i2sb_orthogonal_lowanchor065_k070_e3": "I2SB low-anchor .65",
    "style_covariant_probe_gaussian_low050e9": "Low050 noise iso",
    "style_covariant_probe_stylecov_low050e9": "Low050 noise cov",
}

LABEL_ALLOWLIST = {
    "Seedream",
    "SaMAM style",
    "SaMAM LPIPS",
    "SaMST e5",
    "SaMST e15",
    "e3 best LPIPS",
    "SDE s0.08 ceiling",
    "LatAff s0.45",
    "LatAff s0.75",
    "I2SB e1",
    "I2SB e2",
    "Carrier stop",
    "LowAnc e1",
}

LABEL_OFFSETS = {
    "Seedream": (-70.0, 14.0),
    "IDT": (-34.0, 22.0),
    "SaMAM style": (-56.0, 12.0),
    "SaMAM LPIPS": (10.0, -18.0),
    "SaMST e5": (-68.0, 20.0),
    "SaMST e15": (-66.0, 14.0),
    "e3 best LPIPS": (18.0, -18.0),
    "SDE s0.08 ceiling": (14.0, 24.0),
    "LatAff s0.45": (38.0, 8.0),
    "LatAff s0.75": (18.0, -18.0),
    "I2SB e1": (-52.0, 12.0),
    "I2SB e2": (-52.0, -16.0),
    "Carrier stop": (24.0, -34.0),
}


plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif"],
        "font.size": 9.2,
        "axes.labelsize": 9.5,
        "axes.titlesize": 10.4,
        "xtick.labelsize": 8.0,
        "ytick.labelsize": 8.0,
        "legend.fontsize": 7.0,
        "legend.frameon": False,
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.035,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.22,
        "grid.linewidth": 0.6,
    }
)


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return [dict(row) for row in csv.DictReader(f)]


def _safe_float(value: object) -> float | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def _safe_savefig(fig: plt.Figure, path: Path) -> None:
    try:
        fig.savefig(path)
    except PermissionError as exc:
        print(f"skip locked figure output: {path} ({exc})")


def _fmt(value: float | int | str | None) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return f"{float(value):.12g}"


def _make_row(
    *,
    point_id: str,
    source_group: str,
    family: str,
    variant: str,
    label: str,
    trace_id: str,
    step_or_epoch: str,
    clip_style: float,
    content_lpips: float,
    train_min: float | None = None,
    train_time_sec: float | None = None,
    label_dx: float = 0.0,
    label_dy: float = 0.0,
    note: str = "",
    source_summary: str = "",
) -> dict[str, str]:
    return {
        "point_id": point_id,
        "dataset_surface": DATASET_SURFACE,
        "scope": "transfer",
        "source_group": source_group,
        "family": family,
        "variant": variant,
        "label": label,
        "trace_id": trace_id,
        "step_or_epoch": step_or_epoch,
        "clip_style": _fmt(clip_style),
        "content_lpips": _fmt(content_lpips),
        "one_minus_lpips": _fmt(1.0 - content_lpips),
        "style_minus_idt": _fmt(clip_style - IDT_TRANSFER_CLIP),
        "train_min": _fmt(train_min),
        "train_time_sec": _fmt(train_time_sec),
        "label_dx": _fmt(label_dx),
        "label_dy": _fmt(label_dy),
        "note": note,
        "source_summary": source_summary,
    }


def _phase2_rows() -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for row in _read_csv(PHASE2_POINTS_CSV):
        if row.get("scope") != "transfer":
            continue
        clip = _safe_float(row.get("clip_style"))
        lpips = _safe_float(row.get("content_lpips"))
        if clip is None or lpips is None:
            continue
        rows.append(
            _make_row(
                point_id=f"phase2::{row.get('point_id', '')}",
                source_group="ours_phase2_full_notest",
                family=row.get("family") or "Phase2",
                variant=row.get("variant") or "",
                label=row.get("label") or "",
                trace_id=row.get("trace_id") or row.get("variant") or "phase2",
                step_or_epoch=row.get("step_or_epoch") or "",
                clip_style=clip,
                content_lpips=lpips,
                train_min=_safe_float(row.get("train_min")),
                train_time_sec=_safe_float(row.get("train_time_sec")),
                label_dx=_safe_float(row.get("label_dx")) or 0.0,
                label_dy=_safe_float(row.get("label_dy")) or 0.0,
                note=row.get("note") or "Phase2 run evaluated on current Distinct5-512 classview test.",
                source_summary=row.get("source_summary") or str(PHASE2_POINTS_CSV),
            )
        )
    return rows


def _samam_rows() -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for row in _read_csv(SAMAM_CURVE_CSV):
        step = str(row.get("step") or "").strip()
        clip = _safe_float(row.get("transfer_clip_style"))
        lpips = _safe_float(row.get("transfer_lpips"))
        if not step or clip is None or lpips is None:
            continue
        label = ""
        dx, dy = 0.0, 0.0
        if step == "5750":
            label, dx, dy = "SaMAM style", -52.0, 12.0
        elif step == "19500":
            label, dx, dy = "SaMAM LPIPS", 8.0, -15.0
        rows.append(
            _make_row(
                point_id=f"samam_wikiarts5::{step}",
                source_group="baseline_wikiarts5",
                family="SaMAM",
                variant="patch8_segmented",
                label=label,
                trace_id="samam_wikiarts5_patch8",
                step_or_epoch=f"step_{int(step):05d}",
                clip_style=clip,
                content_lpips=lpips,
                label_dx=dx,
                label_dy=dy,
                note="SaMAM patch=8 segmented reproduction on wikiarts5 full-notest training set.",
                source_summary=row.get("eval_dir") or str(SAMAM_CURVE_CSV),
            )
        )
    return rows


def _samst_rows() -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for row in _read_csv(SAMST_CURVE_CSV):
        epoch = str(row.get("epoch") or "").strip()
        clip = _safe_float(row.get("transfer_clip_style"))
        lpips = _safe_float(row.get("transfer_content_lpips"))
        if not epoch or clip is None or lpips is None:
            continue
        label = {"epoch_0005": "SaMST e5", "epoch_0015": "SaMST e15"}.get(epoch, "")
        dx, dy = (10.0, 10.0) if epoch == "epoch_0005" else (10.0, -14.0)
        rows.append(
            _make_row(
                point_id=f"samst_wikiarts5::{epoch}",
                source_group="baseline_wikiarts5",
                family="SaMST",
                variant="wsl_style_cycle",
                label=label,
                trace_id="samst_wikiarts5",
                step_or_epoch=epoch,
                clip_style=clip,
                content_lpips=lpips,
                train_min=None,
                train_time_sec=None,
                label_dx=dx if label else 0.0,
                label_dy=dy if label else 0.0,
                note="SaMST WSL reproduction on wikiarts5 full-notest training set; not formally converged yet.",
                source_summary=row.get("summary_path") or str(SAMST_CURVE_CSV),
            )
        )
    return rows


def rebuild_points_csv() -> list[dict[str, str]]:
    rows = [
        _make_row(
            point_id="idt_transfer_noop",
            source_group="reference_test_only",
            family="IDT",
            variant="noop_transfer",
            label="IDT",
            trace_id="idt",
            step_or_epoch="0",
            clip_style=IDT_TRANSFER_CLIP,
            content_lpips=0.0,
            train_min=0.0,
            train_time_sec=0.0,
            label_dx=-26.0,
            label_dy=8.0,
            note="No-op source copied to targets; test-only reference.",
            source_summary="SchrodingerBridge/docs/experiments/distinct5_512_20260602/tables/clip_style_vs_1lpips_full_transfer_points.csv",
        ),
        _make_row(
            point_id="seedream45_test_only",
            source_group="external_test_only",
            family="Seedream",
            variant="seedream_4p5_repaired750",
            label="Seedream",
            trace_id="seedream_test_only",
            step_or_epoch="test_only",
            clip_style=0.6920,
            content_lpips=0.4923,
            label_dx=-68.0,
            label_dy=12.0,
            note="External test-only reference on the same Distinct5 classview test; no training-set dependency.",
            source_summary="Related_Works/baseline_pipeline/results/seedream45_api/distinct5_512_seedream45_windhub_20260607",
        ),
    ]
    rows.extend(_samam_rows())
    rows.extend(_samst_rows())
    rows.extend(_phase2_rows())
    POINTS_CSV.parent.mkdir(parents=True, exist_ok=True)
    with POINTS_CSV.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    return rows


def _style_for(trace_id: str) -> tuple[str, str, float, float]:
    return TRACE_STYLES.get(trace_id, ("#E08E00", "P", 46, 0.86))


def _sort_key(row: dict[str, str]) -> tuple[str, float]:
    step = row.get("step_or_epoch") or ""
    number = _safe_float("".join(ch for ch in step if ch.isdigit()))
    return row.get("trace_id") or "", number if number is not None else 0.0


def _xy(row: dict[str, str]) -> tuple[float, float]:
    return float(row["one_minus_lpips"]), float(row["style_minus_idt"])


def _dominates(a: dict[str, str], b: dict[str, str]) -> bool:
    ax, ay = _xy(a)
    bx, by = _xy(b)
    return ax >= bx and ay >= by and (ax > bx or ay > by)


def _pareto_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    frontier: list[dict[str, str]] = []
    for row in rows:
        if any(_dominates(other, row) for other in rows if other is not row):
            continue
        frontier.append(row)
    return sorted(frontier, key=lambda item: _xy(item))


def _normalized(value: float, lo: float, hi: float) -> float:
    span = max(hi - lo, 1e-8)
    return (value - lo) / span


def _distance_to_frontier(
    row: dict[str, str],
    frontier: list[dict[str, str]],
    *,
    x_lo: float,
    x_hi: float,
    y_lo: float,
    y_hi: float,
) -> float:
    if not frontier:
        return 0.0
    x, y = _xy(row)
    x_n = _normalized(x, x_lo, x_hi)
    y_n = _normalized(y, y_lo, y_hi)
    return min(
        ((_normalized(fx, x_lo, x_hi) - x_n) ** 2 + (_normalized(fy, y_lo, y_hi) - y_n) ** 2) ** 0.5
        for fx, fy in (_xy(item) for item in frontier)
    )


def _series_representative(
    rows: list[dict[str, str]],
    global_frontier: list[dict[str, str]],
    *,
    x_lo: float,
    x_hi: float,
    y_lo: float,
    y_hi: float,
) -> dict[str, str]:
    pareto = _pareto_rows(rows)
    if len(pareto) == 1:
        return pareto[0]
    if len(pareto) >= 3:
        ordered = sorted(pareto, key=lambda item: _xy(item))
        start_x, start_y = _xy(ordered[0])
        end_x, end_y = _xy(ordered[-1])
        sx = _normalized(start_x, x_lo, x_hi)
        sy = _normalized(start_y, y_lo, y_hi)
        ex = _normalized(end_x, x_lo, x_hi)
        ey = _normalized(end_y, y_lo, y_hi)
        line_dx = ex - sx
        line_dy = ey - sy
        denom = max((line_dx * line_dx + line_dy * line_dy) ** 0.5, 1e-8)
        scored = []
        for row in ordered[1:-1]:
            x, y = _xy(row)
            xn = _normalized(x, x_lo, x_hi)
            yn = _normalized(y, y_lo, y_hi)
            knee = abs(line_dx * (sy - yn) - (sx - xn) * line_dy) / denom
            frontier_dist = _distance_to_frontier(row, global_frontier, x_lo=x_lo, x_hi=x_hi, y_lo=y_lo, y_hi=y_hi)
            balance = _normalized(x, x_lo, x_hi) + _normalized(y, y_lo, y_hi)
            scored.append((knee, -frontier_dist, balance, row))
        if scored:
            scored.sort(reverse=True, key=lambda item: (item[0], item[1], item[2]))
            if scored[0][0] > 0.015:
                return scored[0][3]
    candidates = []
    for row in pareto:
        frontier_dist = _distance_to_frontier(row, global_frontier, x_lo=x_lo, x_hi=x_hi, y_lo=y_lo, y_hi=y_hi)
        balance = _normalized(_xy(row)[0], x_lo, x_hi) + _normalized(_xy(row)[1], y_lo, y_hi)
        candidates.append((frontier_dist, -balance, row))
    candidates.sort(key=lambda item: (item[0], item[1]))
    return candidates[0][2]


def annotate(ax, row: dict[str, str]) -> None:
    text = (row.get("label") or "").strip()
    if not text or text not in LABEL_ALLOWLIST:
        return
    x, y = _xy(row)
    default_dx = _safe_float(row.get("label_dx")) or 8.0
    default_dy = _safe_float(row.get("label_dy")) or 10.0
    dx, dy = LABEL_OFFSETS.get(text, (default_dx, default_dy))
    color, _, _, _ = _style_for(row.get("trace_id") or "")
    ax.annotate(
        text,
        (x, y),
        xytext=(dx, dy),
        textcoords="offset points",
        fontsize=6.2,
        fontweight="bold",
        color=color,
        arrowprops=dict(arrowstyle="-", color=color, lw=0.5, shrinkA=2, shrinkB=3),
        path_effects=[pe.withStroke(linewidth=2.4, foreground="white")],
        zorder=10,
    )


def plot(points: list[dict[str, str]]) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(5.9, 3.55))
    fig.subplots_adjust(left=0.12, right=0.985, bottom=0.14, top=0.89)
    ax.set_facecolor("#FCFBF8")
    ax.axhspan(-0.08, 0.0, color="#F2E8F7", alpha=0.28, zorder=0)
    ax.axhline(0.0, color="#8E63C0", lw=1.4, ls=(0, (7, 4)), zorder=1)
    plotted_points = [row for row in points if (row.get("trace_id") or "") != "idt"]
    x_vals = [_xy(row)[0] for row in plotted_points]
    y_vals = [_xy(row)[1] for row in plotted_points]
    x_left = max(0.32, min(x_vals) - 0.025) if x_vals else 0.48
    x_right = 0.80
    y_low = min(-0.082, min(y_vals) - 0.014) if y_vals else -0.082
    y_span = max(max(y_vals) - min(y_vals), 0.03) if y_vals else 0.03
    y_high = max(0.074, max(y_vals) + max(0.012, 0.16 * y_span)) if y_vals else 0.074
    ax.text(x_right - 0.006, -0.008, "IDT floor", color="#8E63C0", fontsize=8.2, ha="right", weight="bold")

    by_trace: dict[str, list[dict[str, str]]] = {}
    for row in plotted_points:
        by_trace.setdefault(row.get("trace_id") or "unknown", []).append(row)

    global_frontier = _pareto_rows(plotted_points)
    frontier_ids = {row["point_id"] for row in global_frontier}
    representatives: dict[str, dict[str, str]] = {}
    for trace_id, rows in by_trace.items():
        representatives[trace_id] = _series_representative(
            rows,
            global_frontier,
            x_lo=x_left,
            x_hi=x_right,
            y_lo=y_low,
            y_hi=y_high,
        )

    frontier_xs = [_xy(row)[0] for row in global_frontier]
    frontier_ys = [_xy(row)[1] for row in global_frontier]
    frontier_line, = ax.plot(
        frontier_xs,
        frontier_ys,
        color="#111827",
        lw=1.25,
        alpha=0.82,
        zorder=3,
        label="Current Pareto frontier",
    )

    legend_handles = []
    legend_labels = []
    legend_handles.append(frontier_line)
    legend_labels.append("Current Pareto frontier")

    for trace_id, rows in sorted(by_trace.items()):
        rows = sorted(rows, key=_sort_key)
        xs = [_xy(row)[0] for row in rows]
        ys = [_xy(row)[1] for row in rows]
        color, marker, size, alpha = _style_for(trace_id)
        rep = representatives[trace_id]
        rep_x, rep_y = _xy(rep)
        if len(rows) > 1 and trace_id not in {"idt", "seedream_test_only"}:
            ax.plot(xs, ys, color=color, lw=0.8, alpha=min(0.26, alpha * 0.38), zorder=2)
        bg_scatter_kwargs = {
            "s": max(size * 0.55, 18),
            "c": color,
            "marker": marker,
            "linewidths": 0.4,
            "alpha": min(0.22, alpha * 0.28),
            "zorder": 3,
        }
        if marker != "x":
            bg_scatter_kwargs["edgecolors"] = "white"
        ax.scatter(xs, ys, **bg_scatter_kwargs)

        rep_scatter_kwargs = {
            "s": max(size * 1.2, 54),
            "c": color,
            "marker": marker,
            "linewidths": 1.0,
            "alpha": min(1.0, max(0.88, alpha)),
            "zorder": 7,
        }
        if marker != "x":
            rep_scatter_kwargs["edgecolors"] = "white"
        rep_scatter = ax.scatter([rep_x], [rep_y], **rep_scatter_kwargs)

        if rep["point_id"] in frontier_ids:
            ax.scatter(
                [rep_x],
                [rep_y],
                s=max(size * 1.32, 62),
                facecolors="none",
                edgecolors="#111827",
                linewidths=0.8,
                alpha=0.72,
                zorder=8,
            )

        if trace_id in LEGEND_MAP:
            legend_handles.append(rep_scatter)
            legend_labels.append(LEGEND_MAP[trace_id])

    for row in plotted_points:
        is_rep = representatives[row.get("trace_id") or "unknown"]["point_id"] == row["point_id"]
        is_frontier = row["point_id"] in frontier_ids
        keep_key_baseline_label = (row.get("label") or "") in {"Seedream", "SaMAM style", "SaMAM LPIPS"}
        if keep_key_baseline_label or is_rep or is_frontier:
            annotate(ax, row)

    ax.text(
        0.505,
        -0.071,
        "old distinct5-512 / 1000-per-style points removed",
        color="#7C3AED",
        fontsize=6.5,
        weight="bold",
        alpha=0.78,
    )
    ax.set_title("WikiArt-5 Full-Train Surface: CLIP-S vs. LPIPS", pad=7, fontsize=10.2, fontweight="bold")
    ax.set_xlabel(r"$1-\mathrm{LPIPS}$ $\uparrow$")
    ax.set_ylabel(r"$\Delta_{\mathrm{IDT,tr}}$ (transfer CLIP-S) $\uparrow$")
    ax.set_xlim(x_left, x_right)
    ax.set_ylim(y_low, y_high)
    ax.legend(
        legend_handles,
        legend_labels,
        loc="lower right",
        bbox_to_anchor=(0.985, 0.02),
        borderaxespad=0.0,
        ncol=2,
        columnspacing=0.9,
        handletextpad=0.3,
    )
    _safe_savefig(fig, OUT_DIR / "fig_wikiart5_page1_summary.pdf")
    _safe_savefig(fig, OUT_DIR / "fig_wikiart5_page1_summary.png")
    _safe_savefig(fig, OUT_DIR / "fig_distinct5_page1_summary.pdf")
    _safe_savefig(fig, OUT_DIR / "fig_distinct5_page1_summary.png")
    _safe_savefig(fig, OUT_DIR / "fig_distinct5_page1_summary_clip_delta_idt.pdf")
    _safe_savefig(fig, OUT_DIR / "fig_distinct5_page1_summary_clip_delta_idt.png")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Render the filtered WikiArt-5 homepage CLIP-S/LPIPS figure.")
    parser.add_argument("--rebuild", action="store_true", help="Rebuild the fixed points CSV from current eval artifacts before plotting.")
    args = parser.parse_args()
    if args.rebuild or not POINTS_CSV.exists():
        points = rebuild_points_csv()
    else:
        points = _read_csv(POINTS_CSV)
    plot(points)
    print(f"points={len(points)} csv={POINTS_CSV}")
    print(f"png={OUT_DIR / 'fig_wikiart5_page1_summary.png'}")


if __name__ == "__main__":
    main()
