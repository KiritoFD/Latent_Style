from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


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
    "samam": "#2F7DB7",
    "samst": "#2CA02C",
    "latent_samam": "#0F766E",
    "latent_samst": "#7C3AED",
    "compact": "#D64045",
    "structot": "#B45309",
    "ps": "#9A3412",
    "psv2": "#1D4ED8",
    "seedream": "#C0840A",
    "bg": "#CFCFCF",
    "text": "#333333",
}


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
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


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


def annotate(ax, x: float, y: float, text: str, dx: float, dy: float, color: str) -> None:
    ax.annotate(
        text,
        (x, y),
        xytext=(dx, dy),
        textcoords="offset points",
        fontsize=7.0,
        color=color,
        bbox=dict(boxstyle="round,pad=0.18", fc="white", ec=color, lw=0.55, alpha=0.92),
        arrowprops=dict(arrowstyle="-", color=color, lw=0.55, shrinkA=2, shrinkB=3),
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


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    idt = transfer_row("Reference", "No-op transfer")
    idt["delta_idt_tr"] = 0.0
    samam = transfer_row("SaMAM", "SaMAM 2250")
    samam["delta_idt_tr"] = samam["clip_style"] - idt["clip_style"]
    samst = transfer_row("SaMST", "SaMST e15")
    samst["delta_idt_tr"] = samst["clip_style"] - idt["clip_style"]
    seedream = {
        "label": "Seedream-4.5",
        "clip_style": 0.6920,
        "lpips": 0.4923,
        "one_minus_lpips": 1.0 - 0.4923,
        "delta_idt_tr": 0.6920 - float(idt["clip_style"]),
    }

    compact = best_row("best_compact_mainline_anchor")
    knee = best_row("best_promoted_lpips_ge_070")
    psv2 = best_row("style_best_current")
    latent_samam = latent_curve_row("SaMAM-latent", "Lat SaMAM 1500")
    latent_samst = latent_curve_row("SaMST-latent", "Lat SaMST 1050")

    fig, axes = plt.subplots(1, 2, figsize=(7.35, 2.82), gridspec_kw={"width_ratios": [1.16, 0.84]})

    ax = axes[0]
    ax.set_facecolor("#FCFBF8")
    bg = read_background_points()
    if bg:
        ax.scatter(
            [row["x"] for row in bg],
            [row["y"] for row in bg],
            s=11,
            c=COLORS["bg"],
            alpha=0.22,
            linewidths=0,
            zorder=1,
        )

    ax.axhline(0.0, color=COLORS["idt"], lw=1.2, ls=(0, (7, 4)), zorder=2)
    ax.text(0.82, -0.018, "IDT floor", color=COLORS["idt"], fontsize=8.8, ha="center", weight="bold")

    selected = [
        ("SaMAM-2250", samam, COLORS["samam"], "o"),
        ("SaMST e15", samst, COLORS["samst"], "s"),
        ("Seedream-4.5", seedream, COLORS["seedream"], "P"),
        ("LBM-K", compact, COLORS["compact"], "D"),
        ("LBM-Knee", knee, COLORS["structot"], "o"),
        ("Lat SaMAM", latent_samam, COLORS["latent_samam"], "^"),
        ("Lat SaMST", latent_samst, COLORS["latent_samst"], "v"),
        ("LBM-PS-v2", psv2, COLORS["psv2"], "*"),
    ]
    for label, row, color, marker in selected:
        ax.scatter(
            [row["one_minus_lpips"]],
            [row["delta_idt_tr"]],
            s=72 if marker != "*" else 110,
            c=color,
            marker=marker,
            edgecolors="white",
            linewidths=0.8,
            zorder=4,
        )

    annotate(ax, samam["one_minus_lpips"], samam["delta_idt_tr"], "SaMAM-2250", 10, 8, COLORS["samam"])
    annotate(ax, samst["one_minus_lpips"], samst["delta_idt_tr"], "SaMST e15", -66, -18, COLORS["samst"])
    annotate(ax, seedream["one_minus_lpips"], seedream["delta_idt_tr"], "Seedream-4.5", -80, 8, COLORS["seedream"])
    annotate(ax, compact["one_minus_lpips"], compact["delta_idt_tr"], "LBM-K", 14, 20, COLORS["compact"])
    annotate(ax, knee["one_minus_lpips"], knee["delta_idt_tr"], "LBM-Knee", 20, 22, COLORS["structot"])
    annotate(ax, latent_samam["one_minus_lpips"], latent_samam["delta_idt_tr"], "Lat SaMAM", -54, -8, COLORS["latent_samam"])
    annotate(ax, latent_samst["one_minus_lpips"], latent_samst["delta_idt_tr"], "Lat SaMST", 12, -24, COLORS["latent_samst"])
    annotate(ax, psv2["one_minus_lpips"], psv2["delta_idt_tr"], "LBM-PS-v2", -32, -2, COLORS["psv2"])

    ax.set_title("(a) IDT-calibrated frontier", pad=4)
    ax.set_xlabel(r"$1-\mathrm{LPIPS}$ $\uparrow$")
    ax.set_ylabel(r"$\Delta_{\mathrm{IDT,tr}}$ (transfer CLIP-S) $\uparrow$")
    ax.set_xlim(0.14, 0.85)
    ax.set_ylim(-0.11, 0.102)

    ax = axes[1]
    ax.set_facecolor("#FCFBF8")
    idt_artfid = artfid_row("idt", "idt", scope="full")
    samam_artfid = artfid_row("SaMAM", "SaMAM best-lpips (2250)", scope="full")
    samst_artfid = artfid_row("SaMST", "SaMST e15", scope="full")
    knee_artfid = {"label": "LBM-Knee", "artfid": json_artfid(KNEE_ARTFID_JSON, scope="full")}
    seedream_artfid = {"label": "Seedream-4.5", "artfid": json_artfid(SEEDREAM_ARTFID_JSON, scope="full")}
    art_rows = [
        ("IDT", idt_artfid, COLORS["idt"]),
        ("SaMAM", samam_artfid, COLORS["samam"]),
        ("LBM-Knee", knee_artfid, COLORS["structot"]),
        ("SaMST", samst_artfid, COLORS["samst"]),
        ("Seedream", seedream_artfid, COLORS["seedream"]),
    ]
    xs = np.arange(len(art_rows))
    vals = [row["artfid"] for _, row, _ in art_rows]
    colors = [color for _, _, color in art_rows]
    ax.bar(xs, vals, color=colors, edgecolor="white", linewidth=0.9, zorder=3)
    ax.set_xticks(xs, ["IDT", "SaMAM", "LBM-\nKnee", "SaMST", "Seedream\n4.5"])
    ax.set_ylabel("tw-ArtFID")
    ax.set_title("(b) All-pairs artifact check", pad=4)
    inside_labels = ["IDT", "7.6h", "6.4m", "5.8h", "API"]
    for x, val, txt in zip(xs, vals, inside_labels):
        ax.text(
            x,
            val * 0.52,
            txt,
            ha="center",
            va="center",
            fontsize=8.1 if txt == "IDT" else 7.8,
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
            fontsize=7.4,
            color=COLORS["text"],
            weight="bold",
            zorder=4,
        )

    fig.savefig(OUT_DIR / "fig_distinct5_page1_summary.pdf")
    fig.savefig(OUT_DIR / "fig_distinct5_page1_summary.png")


if __name__ == "__main__":
    main()
