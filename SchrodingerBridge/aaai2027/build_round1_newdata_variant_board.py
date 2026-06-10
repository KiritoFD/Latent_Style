from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patheffects as pe


ROOT = Path(__file__).resolve().parent
WORKSPACE = ROOT.parent.parent
OUT_DIR = ROOT / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

IDT_TRANSFER = 0.6399224616587162

WIKIARTS5_SUMMARY_CSV = ROOT / "wikiarts5_page1" / "wikiarts5_page1_summary.csv"
WIKIARTS5_CURVE_CSV = WORKSPACE / "Related_Works" / "baseline_pipeline" / "results" / "samam_wikiarts5_patch8_segmented_20260610_094447" / "curve_metrics.csv"
ATTNSA_CURVE_CSV = ROOT / "round1_attn_sa_mod_fast_local" / "full_eval_fast_local" / "clip_lpips_curve.csv"
GATED_CURVE_CSV = ROOT / "round1_attn_gated_spade_remote_full_eval_pull" / "clip_lpips_curve.csv"
POINTS_CSV = ROOT / "round1_newdata_variant_board.csv"

OUT_PNG = OUT_DIR / "fig_round1_newdata_variant_board.png"
OUT_PDF = OUT_DIR / "fig_round1_newdata_variant_board.pdf"


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))

def _annotate(ax, x: float, y: float, text: str, dx: float, dy: float, color: str) -> None:
    ax.annotate(
        text,
        (x, y),
        xytext=(dx, dy),
        textcoords="offset points",
        fontsize=7.2,
        fontweight="semibold",
        color=color,
        arrowprops=dict(arrowstyle="-", color=color, lw=0.6, shrinkA=2, shrinkB=3),
        path_effects=[pe.withStroke(linewidth=2.2, foreground="white")],
    )


def _find_row(rows: list[dict[str, str]], label: str) -> dict[str, str]:
    for row in rows:
        if str(row.get("label", "")).strip() == label:
            return row
    raise KeyError(label)


def _best(rows: list[dict[str, str]], *, style_key: str, lpips_key: str, mode: str) -> dict[str, str]:
    if mode == "style":
        return max(rows, key=lambda r: (float(r[style_key]), -float(r[lpips_key])))
    if mode == "lpips":
        return min(rows, key=lambda r: (float(r[lpips_key]), -float(r[style_key])))
    raise ValueError(mode)


def _latest(rows: list[dict[str, str]], *, epoch_key: str) -> dict[str, str]:
    return max(rows, key=lambda r: int("".join(ch for ch in str(r[epoch_key]) if ch.isdigit()) or "-1"))


def main() -> int:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "DejaVu Serif"],
            "font.size": 9.0,
            "axes.labelsize": 9.2,
            "axes.titlesize": 10.2,
            "xtick.labelsize": 7.8,
            "ytick.labelsize": 7.8,
            "legend.fontsize": 7.2,
            "figure.dpi": 300,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.03,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.23,
            "grid.linewidth": 0.55,
        }
    )

    wikiarts5_rows = _read_csv(WIKIARTS5_SUMMARY_CSV)
    wikiarts5_curve_rows = _read_csv(WIKIARTS5_CURVE_CSV)
    attnsa_rows = _read_csv(ATTNSA_CURVE_CSV)
    gated_rows = _read_csv(GATED_CURVE_CSV)
    point_rows = _read_csv(POINTS_CSV)

    # Keep this sanity check so the fixed point CSV cannot silently drift away
    # from the manually named summary rows.
    _find_row(wikiarts5_rows, "WikiArts5 best-CLIP")
    _find_row(wikiarts5_rows, "WikiArts5 best-LPIPS")
    _find_row(wikiarts5_rows, "WikiArts5 latest")
    _best(attnsa_rows, style_key="transfer_clip_style", lpips_key="transfer_content_lpips", mode="style")
    _best(attnsa_rows, style_key="transfer_clip_style", lpips_key="transfer_content_lpips", mode="lpips")
    _latest(attnsa_rows, epoch_key="epoch")
    _best(gated_rows, style_key="transfer_clip_style", lpips_key="transfer_content_lpips", mode="style")
    _best(gated_rows, style_key="transfer_clip_style", lpips_key="transfer_content_lpips", mode="lpips")
    _latest(gated_rows, epoch_key="epoch")

    fig, left = plt.subplots(1, 1, figsize=(6.2, 4.9), dpi=300)
    left.set_facecolor("#FCFBF8")
    left.axhspan(-0.12, 0.0, color="#F2E8F7", alpha=0.30, zorder=0)
    left.axhline(0.0, color="#8E63C0", lw=1.5, ls=(0, (7, 4)), zorder=1)
    left.text(0.786, -0.013, "IDT floor", color="#8E63C0", fontsize=8.2, ha="right", weight="bold")

    samam_curve_x = [1.0 - float(row["transfer_lpips"]) for row in wikiarts5_curve_rows]
    samam_curve_y = [float(row["transfer_clip_style"]) - IDT_TRANSFER for row in wikiarts5_curve_rows]
    left.plot(samam_curve_x, samam_curve_y, color="#F07F5A", linewidth=1.6, alpha=0.9, zorder=2, label="W5 SaMAM trajectory")
    left.scatter(samam_curve_x, samam_curve_y, s=18, color="#F0A085", alpha=0.60, linewidths=0, zorder=2)

    attnsa_curve_x = [1.0 - float(row["transfer_content_lpips"]) for row in attnsa_rows]
    attnsa_curve_y = [float(row["transfer_clip_style"]) - IDT_TRANSFER for row in attnsa_rows]
    left.plot(attnsa_curve_x, attnsa_curve_y, color="#1D4ED8", linewidth=1.25, alpha=0.75, zorder=2, label="AttnSA trajectory")
    left.scatter(attnsa_curve_x, attnsa_curve_y, s=14, color="#6E93FF", alpha=0.40, linewidths=0, zorder=2)

    gated_curve_x = [1.0 - float(row["transfer_content_lpips"]) for row in gated_rows]
    gated_curve_y = [float(row["transfer_clip_style"]) - IDT_TRANSFER for row in gated_rows]
    left.plot(gated_curve_x, gated_curve_y, color="#16A085", linewidth=1.25, alpha=0.78, zorder=2, label="GatedSPADE trajectory")
    left.scatter(gated_curve_x, gated_curve_y, s=14, color="#52C7B0", alpha=0.40, linewidths=0, zorder=2)

    def add_point(row: dict[str, str]) -> None:
        clip = float(row["transfer_clip_style"])
        lpips = float(row["transfer_lpips"])
        x = 1.0 - lpips
        y = clip - IDT_TRANSFER
        size = float(row.get("size", "76"))
        color = row["color"]
        marker = row["marker"]
        label = row["label"]
        dx = float(row["dx"])
        dy = float(row["dy"])
        left.scatter([x], [y], s=size, c=color, marker=marker, edgecolors="white", linewidths=0.9, zorder=4)
        _annotate(left, x, y, label, dx=dx, dy=dy, color=color)

    for point_row in point_rows:
        add_point(point_row)

    left.set_title("New-data variant board on the fixed test split")
    left.set_xlabel(r"$1 - \mathrm{LPIPS}_{tr}$")
    left.set_ylabel(r"$\Delta_{\mathrm{IDT,tr}}$ (CLIP-S) $\uparrow$")
    left.set_xlim(0.35, 0.79)
    left.set_ylim(-0.11, 0.07)
    left.legend(loc="lower left")

    fig.savefig(OUT_PNG)
    fig.savefig(OUT_PDF)
    plt.close(fig)
    print(OUT_PNG)
    print(POINTS_CSV)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
