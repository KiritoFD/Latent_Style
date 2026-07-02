from __future__ import annotations

import csv
import json
import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patheffects as pe


ROOT = Path(__file__).resolve().parent
WORKSPACE = ROOT.parent.parent
EXPERIMENTS = ROOT.parent / "docs" / "experiments"
BEST_CSV = WORKSPACE / "best.csv"
TRANSFER_CSV = (
    EXPERIMENTS
    / "distinct5_512_20260602"
    / "tables"
    / "clip_style_vs_1lpips_full_transfer_points.csv"
)
ARTFID_CSV = (
    EXPERIMENTS
    / "comparison_20260602"
    / "artfid_comparison_points.csv"
)
ALL_POINTS_CSV = ROOT / "fig_distinct5_all_points_big.csv"
INTROSTYLE_PAGE1_CSV = ROOT / "introstyle_page1" / "introstyle_page1_summary.csv"
PAGE1_ARTFID_RERUN_CSV = ROOT / "page1_bundle" / "page1_artfid_rerun_summary.csv"
PHASE2_POINTS_CSV = EXPERIMENTS / "phase2_fiber_bundle" / "plot_points.csv"
AUX_ARTIFACT_CSV = ROOT / "final" / "distinct5_aux_artifact_table.csv"
DASHBOARD_HTML = ROOT.parent / "exp" / "phase616_live_dashboard" / "phase616_live_dashboard.html"
OUT_DIR = ROOT / "figures"
KNEE_ARTFID_JSON = ROOT / "local_eval" / "lbm_knee_e13_artfid" / "aggregate_targetwise_artfid_fast_repro.json"
SEEDREAM_ARTFID_JSON = (
    ROOT / "local_eval" / "seedream_repaired750_artfid" / "aggregate_targetwise_artfid_fast_repro.json"
)


plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif"],
        "font.size": 9.2,
        "axes.labelsize": 9.4,
        "axes.titlesize": 10.0,
        "xtick.labelsize": 7.7,
        "ytick.labelsize": 7.7,
        "legend.fontsize": 7.0,
        "legend.frameon": False,
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.03,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.24,
        "grid.linewidth": 0.6,
    }
)


COLORS = {
    "idt": "#8E63C0",
    "samam": "#5D8FBF",
    "samst": "#55A85B",
    "latent_samam": "#3C8F89",
    "latent_samst": "#8E66D9",
    "compact": "#D64045",
    "structot": "#B45309",
    "ps": "#9A3412",
    "psv2": "#1D4ED8",
    "seedream": "#C58A2B",
    "phase2": "#E08E00",
    "bg": "#CFCFCF",
    "text": "#333333",
    "lbm_band": "#F3E7D6",
    "stylegallery": "#F97316",
    "styleshot": "#EF4444",
    "csgo_low_vram": "#64748B",
    "fcsb": "#14B8A6",
}


DASHBOARD_PARETO_POINTS = [
    (0.627, 0.0321),
    (0.502, 0.0651),
    (0.494, 0.0751),
    (0.456, 0.0771),
    (0.419, 0.0871),
    (0.398, 0.0531),
    (0.441, 0.0641),
]


FC_SB_FOLLOWUP_POINTS = [
    {
        "label": "I7",
        "x": 1.0 - 0.3625,
        "style_minus_idt": 0.7017 - 0.6399208252628644,
        "color": "#0F766E",
        "dx": -10.0,
        "dy": 12.0,
    },
    {
        "label": "U4",
        "x": 1.0 - 0.3660,
        "style_minus_idt": 0.7225 - 0.6399208252628644,
        "color": "#14B8A6",
        "dx": -2.0,
        "dy": 13.0,
    },
    {
        "label": "V6",
        "x": 1.0 - 0.3722,
        "style_minus_idt": 0.7262 - 0.6399208252628644,
        "color": "#0891B2",
        "dx": 10.0,
        "dy": 12.0,
    },
    {
        "label": "V3",
        "x": 1.0 - 0.3963,
        "style_minus_idt": 0.7295 - 0.6399208252628644,
        "color": "#1D4ED8",
        "dx": 10.0,
        "dy": -14.0,
    },
]

PHASE2_TRACE_STYLES = {
    "k070_e1_e5": ("#E08E00", "P"),
    "pattn_enhanced_tok_e1_e9": ("#0F766E", "X"),
    "pattn_enhanced_tok_e1_e10": ("#0F766E", "X"),
    "fiber_sde_iso_scan": ("#2563EB", "o"),
    "fiber_sde_fiber_scan": ("#DC2626", "D"),
    "rgbcal_k070_e3": ("#6B7280", "h"),
    "appearance_blend_k070_e3": ("#7C3AED", "v"),
    "pc_lowpass_k070_e3": ("#0891B2", "d"),
}


def phase2_trace_style(trace_id: str) -> tuple[str, str]:
    return PHASE2_TRACE_STYLES.get(trace_id, (COLORS["phase2"], "P"))


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


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def load_dashboard_data() -> dict[str, object]:
    text = DASHBOARD_HTML.read_text(encoding="utf-8")
    match = re.search(r"window\.PHASE616_LIVE_DATA\s*=\s*(\{.*?\})\s*;\s*let data =", text, re.S)
    if not match:
        raise RuntimeError(f"Could not extract dashboard payload from {DASHBOARD_HTML}")
    return json.loads(match.group(1))


def transfer_row(family: str, label: str) -> dict[str, object]:
    for row in read_csv(TRANSFER_CSV):
        if row["scope"] == "transfer" and row["family"] == family and row["label"] == label:
            lpips = float(row["content_lpips"])
            clip_style = float(row["clip_style"])
            return {
                "label": label,
                "clip_style": clip_style,
                "lpips": lpips,
                "one_minus_lpips": 1.0 - lpips,
                "delta_idt_tr": None,
            }
    raise KeyError((family, label))


def best_row(slot: str) -> dict[str, object]:
    for row in read_csv(BEST_CSV):
        if row["slot"] == slot:
            return {
                "label": row["experiment"],
                "clip_style": float(row["clip_style"]),
                "lpips": float(row["content_lpips"]),
                "one_minus_lpips": float(row["one_minus_lpips"]),
                "delta_idt_tr": float(row["delta_idt_tr"]),
            }
    raise KeyError(slot)


def artfid_row(method: str, label: str, *, scope: str = "full") -> dict[str, object]:
    for row in read_csv(ARTFID_CSV):
        if row["dataset"] == "distinct5_512" and row["scope"] == scope and row["method"] == method and row["label"] == label:
            return {
                "label": label,
                "artfid": float(row["aggregate_art_fid"]),
            }
    raise KeyError((method, label))


def json_artfid(path: Path, *, scope: str, key: str = "aggregate_art_fid") -> float:
    payload = json.loads(path.read_text(encoding="utf-8"))
    node = payload[scope]
    return float(node[key])


def annotate(
    ax,
    x: float,
    y: float,
    text: str,
    dx: float,
    dy: float,
    color: str,
    *,
    fontsize: float = 6.2,
    arrow: bool = True,
    weight: str = "semibold",
    alpha: float = 1.0,
) -> None:
    ax.annotate(
        text,
        (x, y),
        xytext=(dx, dy),
        textcoords="offset points",
        fontsize=fontsize,
        fontweight=weight,
        color=color,
        alpha=alpha,
        arrowprops=(dict(arrowstyle="-", color=color, lw=0.5, shrinkA=2, shrinkB=3) if arrow else None),
        path_effects=[pe.withStroke(linewidth=2.4, foreground="white")],
    )


def read_background_points() -> list[dict[str, float]]:
    if not ALL_POINTS_CSV.exists():
        return []
    rows = []
    for row in read_csv(ALL_POINTS_CSV):
        lpips = _safe_float(row.get("content_lpips"))
        clip_style = _safe_float(row.get("clip_style"))
        if lpips is None or clip_style is None:
            continue
        if row.get("family") == "Reference":
            continue
        rows.append(
            {
                "x": 1.0 - lpips,
                "y": clip_style - 0.6399208252628644,
            }
        )
    return rows


def latent_curve_row(family: str, label: str) -> dict[str, object]:
    for row in read_csv(ALL_POINTS_CSV):
        if row.get("family") == family and row.get("label") == label:
            lpips = float(row["content_lpips"])
            clip_style = float(row["clip_style"])
            return {
                "label": label,
                "clip_style": clip_style,
                "lpips": lpips,
                "one_minus_lpips": 1.0 - lpips,
                "delta_idt_tr": clip_style - 0.6399208252628644,
            }
    raise KeyError((family, label))


def phase2_transfer_rows(idt_clip: float) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for row in read_csv(PHASE2_POINTS_CSV):
        if row.get("scope") != "transfer":
            continue
        clip_style = _safe_float(row.get("clip_style"))
        lpips = _safe_float(row.get("content_lpips"))
        if clip_style is None or lpips is None:
            continue
        delta = _safe_float(row.get("style_minus_idt"))
        if delta is None:
            delta = clip_style - idt_clip
        rows.append(
            {
                "label": row.get("label", row.get("variant") or row.get("step_or_epoch") or "Phase2"),
                "variant": row.get("variant") or "",
                "trace_id": row.get("trace_id") or row.get("variant") or row.get("label") or "phase2",
                "step_or_epoch": row.get("step_or_epoch") or "",
                "clip_style": clip_style,
                "lpips": lpips,
                "one_minus_lpips": 1.0 - lpips,
                "delta_idt_tr": delta,
                "label_dx": _safe_float(row.get("label_dx")),
                "label_dy": _safe_float(row.get("label_dy")),
            }
        )
    return rows


def page1_artfid_row(label: str) -> dict[str, object] | None:
    for row in read_csv(PAGE1_ARTFID_RERUN_CSV):
        if row.get("label") == label:
            return {
                "label": label,
                "artfid": float(row["aggregate_art_fid"]),
                "train_time_label": row["train_time_label"],
            }
    aux_labels = {
        "LBM-K": "LBM-K e1",
        "LBM-Knee": "LBM-Knee e13",
        "LBM-PS-v2": "LBM-PS-v2 e13",
        "SaMST e15": "SaMST e15",
        "Seedream-4.5": "Seedream-4.5",
    }
    aux_point = aux_labels.get(label)
    if aux_point:
        for row in read_csv(AUX_ARTIFACT_CSV):
            if row.get("point") == aux_point:
                artfid = _safe_float(row.get("tw_artfid_all_down"))
                if artfid is not None:
                    return {"label": label, "artfid": artfid, "train_time_label": ""}
    artfid_specs = {
        "IDT": ("idt", "idt", "ref"),
        "SaMAM-2250": ("SaMAM", "SaMAM best-lpips (2250)", "7.6h"),
        "LBM-K": ("LANCET", "LANCET best-style (K e1)", "1.2m"),
    }
    spec = artfid_specs.get(label)
    if spec is not None:
        method, row_label, train_time_label = spec
        for row in read_csv(ARTFID_CSV):
            if (
                row.get("dataset") == "distinct5_512"
                and row.get("scope") == "full"
                and row.get("method") == method
                and row.get("label") == row_label
            ):
                artfid = _safe_float(row.get("aggregate_art_fid"))
                if artfid is not None:
                    return {"label": label, "artfid": artfid, "train_time_label": train_time_label}
    return None


def introstyle_row(run: str) -> dict[str, object]:
    for row in read_csv(INTROSTYLE_PAGE1_CSV):
        if row.get("run") == run:
            return {
                "label": row.get("plot_label", run),
                "target_style": float(row["transfer_target_style_score"]),
                "style_margin": float(row["transfer_style_margin"]),
                "delta_idt_style": float(row["transfer_delta_idt_style"]),
            }
    raise KeyError(run)


def y_value_for(metric: str, clip_row: dict[str, object], intro_row: dict[str, object] | None) -> float:
    if metric == "clip_delta_idt":
        return float(clip_row["delta_idt_tr"])
    if intro_row is None:
        raise KeyError(f"missing IntroStyle row for metric {metric}")
    if metric == "introstyle_delta_idt":
        return float(intro_row["delta_idt_style"])
    if metric == "introstyle_margin":
        return float(intro_row["style_margin"])
    raise ValueError(metric)


def axis_ylabel(metric: str) -> str:
    if metric == "clip_delta_idt":
        return r"$\Delta_{\mathrm{IDT,tr}}$ (transfer CLIP-S) $\uparrow$"
    if metric == "introstyle_delta_idt":
        return r"$\Delta_{\mathrm{IDT}}$ (IntroStyle target score) $\uparrow$"
    if metric == "introstyle_margin":
        return r"IntroStyle style margin $\uparrow$"
    raise ValueError(metric)


def axis_title(metric: str) -> str:
    if metric == "clip_delta_idt":
        return "(a) Real style direction on the transfer plane"
    if metric == "introstyle_delta_idt":
        return "(a) IntroStyle IDT-calibrated frontier"
    if metric == "introstyle_margin":
        return "(a) IntroStyle specificity frontier"
    raise ValueError(metric)


def axis_ylim(metric: str) -> tuple[float, float]:
    if metric == "clip_delta_idt":
        return (-0.07, 0.19)
    if metric == "introstyle_delta_idt":
        return (-0.24, 0.03)
    if metric == "introstyle_margin":
        return (-0.24, 0.02)
    raise ValueError(metric)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--y-metric",
        choices=["clip_delta_idt", "introstyle_delta_idt", "introstyle_margin"],
        default="clip_delta_idt",
    )
    parser.add_argument(
        "--include-phase2",
        action="store_true",
        help="Overlay phase-2 exploratory points on panel (a). Disabled by default for the paper-facing figure.",
    )
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    dashboard = load_dashboard_data()
    dashboard_baselines = dashboard.get("baselines", {})

    idt = transfer_row("Reference", "No-op transfer")
    idt["delta_idt_tr"] = 0.0
    phase2_rows = phase2_transfer_rows(float(idt["clip_style"])) if args.include_phase2 else []
    samam = transfer_row("SaMAM", "SaMAM 2250")
    samam["delta_idt_tr"] = samam["clip_style"] - idt["clip_style"]
    samst = transfer_row("SaMST", "SaMST e15")
    samst["delta_idt_tr"] = samst["clip_style"] - idt["clip_style"]
    dashboard_seedream = dashboard_baselines.get("seedream", {})
    seedream = {
        "label": "Seedream-4.5",
        "clip_style": float(dashboard_seedream.get("clip_style", 0.6920)),
        "lpips": float(dashboard_seedream.get("lpips", 0.4923)),
        "one_minus_lpips": float(dashboard_seedream.get("x", 1.0 - 0.4923)),
        "delta_idt_tr": float(dashboard_seedream.get("style_minus_idt", 0.6920 - float(idt["clip_style"]))),
    }

    compact = best_row("best_compact_mainline_anchor")
    knee = best_row("best_promoted_lpips_ge_070")
    psv2 = best_row("style_best_current")
    latent_samam = latent_curve_row("SaMAM-latent", "Lat SaMAM 1500")
    latent_samst = latent_curve_row("SaMST-latent", "Lat SaMST 1050")

    intro_rows = {}
    if args.y_metric != "clip_delta_idt":
        for run in [
            "IDT",
            "SaMAM_2250",
            "SaMST_e15",
            "Lat_SaMAM_step1500",
            "Lat_SaMST_batch1050",
            "LBM-K_e1",
            "LBM-Knee_e13",
            "LBM-PS-v2_e13",
            "Seedream_repaired750",
        ]:
            intro_rows[run] = introstyle_row(run)

    fig, axes = plt.subplots(1, 2, figsize=(7.7, 2.9), gridspec_kw={"width_ratios": [1.08, 1.02]})

    ax = axes[0]
    ax.set_facecolor("#FCFBF8")
    ax.axhspan(-0.07, 0.0, color="#F2E8F7", alpha=0.22, zorder=0)
    bg = read_background_points()
    if bg:
        ax.scatter(
            [row["x"] for row in bg],
            [row["y"] for row in bg],
            s=9,
            c=COLORS["bg"],
            alpha=0.10,
            linewidths=0,
            zorder=1,
        )

    ax.axhline(0.0, color=COLORS["idt"], lw=1.6, ls=(0, (7, 4)), zorder=2)
    ax.text(0.41, 0.004, "IDT floor", color=COLORS["idt"], fontsize=7.9, ha="left", weight="bold")
    ax.text(0.60, -0.028, "wrong-direction zone", color=COLORS["samam"], fontsize=5.8, ha="center", weight="bold")

    samam_curve = sorted(
        [
            (float(row["x"]), float(row["style_minus_idt"]))
            for row in dashboard_baselines.get("samam_curve", [])
        ],
        key=lambda p: p[0],
    )
    if samam_curve:
        ax.plot([p[0] for p in samam_curve], [p[1] for p in samam_curve], color=COLORS["samam"], lw=2.4, alpha=0.88, zorder=3)
        ax.scatter([p[0] for p in samam_curve], [p[1] for p in samam_curve], s=24, c=COLORS["samam"], edgecolors="#0f172a", linewidths=0.55, zorder=4)
        samam_main = max(samam_curve, key=lambda p: p[0])
        ax.scatter([samam_main[0]], [samam_main[1]], s=58, c=COLORS["samam"], edgecolors="white", linewidths=0.9, zorder=6)
        annotate(ax, samam_main[0], samam_main[1], "SaMAM\nbelow IDT", 10, 8, COLORS["samam"], fontsize=5.8, weight="bold")

    samst_curve = sorted(
        [
            (float(row["x"]), float(row["style_minus_idt"]))
            for row in dashboard_baselines.get("samst_curve", [])
        ],
        key=lambda p: p[0],
    )
    if samst_curve:
        ax.plot([p[0] for p in samst_curve], [p[1] for p in samst_curve], color=COLORS["samst"], lw=2.1, alpha=0.82, zorder=3)
        ax.scatter([p[0] for p in samst_curve], [p[1] for p in samst_curve], s=22, c=COLORS["samst"], edgecolors="#0f172a", linewidths=0.55, zorder=4)
        samst_main = max(samst_curve, key=lambda p: p[0])
        annotate(ax, samst_main[0], samst_main[1], "SaMST e15", 10, -14, COLORS["samst"], arrow=False, fontsize=5.8, weight="medium", alpha=0.86)

    ax.scatter([seedream["one_minus_lpips"]], [seedream["delta_idt_tr"]], s=48, c=COLORS["seedream"], edgecolors="#0f172a", linewidths=0.8, zorder=5)
    annotate(ax, seedream["one_minus_lpips"], seedream["delta_idt_tr"], "Seedream", -14, -13, COLORS["seedream"], arrow=False, fontsize=5.8, weight="medium", alpha=0.86)

    dashboard_external_points = [
        row for row in dashboard_baselines.get("external_points", [])
        if row.get("id") in {"stylegallery", "styleshot", "csgo_low_vram"}
    ]
    for row in dashboard_external_points:
        color = COLORS[row["id"]]
        ax.scatter([row["x"]], [row["style_minus_idt"]], s=34, c=color, edgecolors="#0f172a", linewidths=0.75, zorder=5)
        annotate(
            ax,
            row["x"],
            row["style_minus_idt"],
            row["label"],
            row["label_dx"],
            row["label_dy"],
            color,
            arrow=False,
            fontsize=5.8,
            weight="medium",
            alpha=0.9,
        )

    pareto_x = [p[0] for p in DASHBOARD_PARETO_POINTS]
    pareto_y = [p[1] for p in DASHBOARD_PARETO_POINTS]
    ax.plot(pareto_x, pareto_y, color=COLORS["fcsb"], lw=1.8, ls=(0, (8, 4)), alpha=0.68, zorder=2)
    ax.text(0.425, 0.095, "Anti-degeneration\nPareto front", color=COLORS["fcsb"], fontsize=5.8, weight="bold", ha="left", va="center")

    fc_x = [p["x"] for p in FC_SB_FOLLOWUP_POINTS]
    fc_y = [p["style_minus_idt"] for p in FC_SB_FOLLOWUP_POINTS]
    ax.plot(fc_x, fc_y, color=COLORS["fcsb"], lw=2.5, zorder=6)
    for point in FC_SB_FOLLOWUP_POINTS:
        ax.scatter([point["x"]], [point["style_minus_idt"]], s=54, c=point["color"], edgecolors="white", linewidths=0.9, zorder=7)
        annotate(
            ax,
            point["x"],
            point["style_minus_idt"],
            point["label"],
            point["dx"],
            point["dy"],
            point["color"],
            fontsize=6.0,
            weight="bold",
        )

    ax.set_title(axis_title(args.y_metric), pad=6, fontsize=9.2, fontweight="bold")
    ax.set_xlabel(r"$1-\mathrm{LPIPS}$ $\uparrow$")
    ax.set_ylabel(axis_ylabel(args.y_metric))
    ax.set_xlim(0.14, 0.85)
    ax.set_ylim(*axis_ylim(args.y_metric))

    ax = axes[1]
    ax.set_facecolor("#FCFBF8")
    idt_artfid = page1_artfid_row("IDT")
    samam_artfid = page1_artfid_row("SaMAM-2250")
    latent_samam_artfid = page1_artfid_row("Lat SaMAM")
    compact_artfid = page1_artfid_row("LBM-K")
    knee_artfid = page1_artfid_row("LBM-Knee")
    samst_artfid = page1_artfid_row("SaMST e15")
    latent_samst_artfid = page1_artfid_row("Lat SaMST")
    seedream_artfid = page1_artfid_row("Seedream-4.5")
    psv2_artfid = page1_artfid_row("LBM-PS-v2")
    raw_art_rows = [
        ("IDT", idt_artfid, COLORS["idt"]),
        ("SaMAM", samam_artfid, COLORS["samam"]),
        ("LBM-\nK", compact_artfid, COLORS["compact"]),
        ("LBM-\nKnee", knee_artfid, COLORS["structot"]),
        ("LBM-\nPS-v2", psv2_artfid, COLORS["psv2"]),
        ("SaMST", samst_artfid, COLORS["samst"]),
        ("Seedream-\n4.5", seedream_artfid, COLORS["seedream"]),
    ]
    art_rows = [(label, row, color) for label, row, color in raw_art_rows if row is not None]
    if not art_rows:
        ax.text(
            0.5,
            0.5,
            "tw-ArtFID source missing",
            transform=ax.transAxes,
            ha="center",
            va="center",
            fontsize=8.0,
            color=COLORS["text"],
        )
        vals = [1.0]
    else:
        xs = np.arange(len(art_rows))
        vals = [row["artfid"] for _, row, _ in art_rows]
        colors = [color for _, _, color in art_rows]
        ax.axvspan(2.5, min(5.5, max(2.5, len(art_rows) - 0.5)), color=COLORS["lbm_band"], alpha=0.36, zorder=0)
        if len(art_rows) >= 4:
            ax.text(4.0, max(vals) * 0.87, "LBM family", ha="center", va="center", fontsize=6.4, color="#9A3412", weight="bold")
        ax.bar(xs, vals, color=colors, edgecolor="white", linewidth=0.9, zorder=3)
        ax.set_xticks(xs, [label for label, _, _ in art_rows])
        ax.tick_params(axis="x", labelsize=6.2)
        inside_labels = []
        for _, row, _ in art_rows:
            txt = str(row["train_time_label"])
            txt = {
                "140.6m": "2.3h",
                "~35m": "35m",
                "ref": "ref",
                "": "",
            }.get(txt, txt)
            inside_labels.append(txt)
        for x, val, txt in zip(xs, vals, inside_labels):
            if txt:
                ax.text(
                    x,
                    val * 0.52,
                    txt,
                    ha="center",
                    va="center",
                    fontsize=6.3 if txt != "ref" else 6.6,
                    color="white",
                    weight="bold",
                    zorder=4,
                )
            ax.text(
                x,
                val + max(vals) * 0.02,
                f"{val:.1f}",
                ha="center",
                va="bottom",
                fontsize=6.1,
                color=COLORS["text"],
                weight="bold",
                zorder=4,
            )
    ax.set_ylabel("tw-ArtFID")
    ax.set_title("(b) All-pairs tw-ArtFID", pad=4, fontsize=9.2, fontweight="bold")

    metric_suffix = {
        "clip_delta_idt": "clip_delta_idt",
        "introstyle_delta_idt": "introstyle_delta_idt",
        "introstyle_margin": "introstyle_margin",
    }[args.y_metric]
    if args.y_metric == "clip_delta_idt":
        fig.savefig(OUT_DIR / "fig_distinct5_page1_summary.pdf")
        fig.savefig(OUT_DIR / "fig_distinct5_page1_summary.png")
    fig.savefig(OUT_DIR / f"fig_distinct5_page1_summary_{metric_suffix}.pdf")
    fig.savefig(OUT_DIR / f"fig_distinct5_page1_summary_{metric_suffix}.png")


if __name__ == "__main__":
    main()
