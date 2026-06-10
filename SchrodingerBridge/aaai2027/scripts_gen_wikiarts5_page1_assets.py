from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parent
WORKSPACE = ROOT.parent.parent
SOURCE_ROOT = WORKSPACE / "Dataset" / "distinct5_512" / "test"
IDT_METRICS = (
    WORKSPACE
    / "SchrodingerBridge"
    / "docs"
    / "experiments"
    / "idt_eval_20260602"
    / "distinct5_512"
    / "idt_5x5"
    / "metrics.csv"
)
OLD_SAMAM_METRICS = ROOT / "page1_bundle" / "samam_2250_full" / "metrics.csv"
OLD_SAMAM_IMAGES = ROOT / "page1_bundle" / "samam_2250_full" / "images"
NEW_SAMAM_ROOT = (
    WORKSPACE
    / "Related_Works"
    / "baseline_pipeline"
    / "results"
    / "samam_wikiarts5_patch8_segmented_20260610_094447"
)
OUT_ROOT = ROOT / "wikiarts5_page1"
FIG_ROOT = ROOT / "figures"

SUMMARY_JSON = OUT_ROOT / "wikiarts5_page1_summary.json"
SUMMARY_CSV = OUT_ROOT / "wikiarts5_page1_summary.csv"
CURVE_CSV = OUT_ROOT / "wikiarts5_page1_curve.csv"
SUMMARY_PNG = FIG_ROOT / "fig_wikiarts5_page1_summary.png"
SUMMARY_PDF = FIG_ROOT / "fig_wikiarts5_page1_summary.pdf"
QUAL_PNG = FIG_ROOT / "fig_wikiarts5_qualitative_main.png"
QUAL_PDF = FIG_ROOT / "fig_wikiarts5_qualitative_main.pdf"

FAIL_CASE = {
    "src_style": "Impressionism",
    "tgt_style": "Minimalism",
    "src_stem": "alfred-sisley_riverbank-at-veneux-1881",
    "row_label": "Impressionism -> Minimalism",
}
GAIN_CASE = {
    "src_style": "Rococo",
    "tgt_style": "Minimalism",
    "src_stem": "antoine-pesne_carl-heinrich-graun",
    "row_label": "Rococo -> Minimalism",
}

CELL = 144
LEFT_W = 256
TOP_PAD = 16
TITLE_H = 32
HEADER_H = 28
ROW_GAP = 14


@dataclass
class EvalStats:
    label: str
    source: str
    step: int | None
    metrics_csv: Path
    images_dir: Path | None
    count_all: int
    count_transfer: int
    count_identity: int
    clip_all: float
    lpips_all: float
    clip_transfer: float
    lpips_transfer: float
    clip_identity: float
    lpips_identity: float

    @property
    def one_minus_lpips_transfer(self) -> float:
        return 1.0 - self.lpips_transfer

    def delta_idt(self, idt_transfer_clip: float) -> float:
        return self.clip_transfer - idt_transfer_clip


def _font(size: int, *, bold: bool = False):
    candidates = [
        "C:/Windows/Fonts/timesbd.ttf" if bold else "C:/Windows/Fonts/times.ttf",
        "C:/Windows/Fonts/georgiab.ttf" if bold else "C:/Windows/Fonts/georgia.ttf",
    ]
    for cand in candidates:
        path = Path(cand)
        if not path.exists():
            continue
        try:
            return ImageFont.truetype(str(path), size)
        except Exception:
            pass
    return ImageFont.load_default()


FONT = _font(17)
FONT_B = _font(18, bold=True)
FONT_ROW = _font(17, bold=True)
FONT_TITLE = _font(22, bold=True)
FONT_META = _font(13)


def _safe_float(value: object) -> float:
    return float(str(value).strip())


def metric_value(row: dict[str, str], key: str) -> float:
    if key in row:
        return _safe_float(row[key])
    if key == "lpips" and "content_lpips" in row:
        return _safe_float(row["content_lpips"])
    raise KeyError(key)


def canonical_src_stem(src_style: str, src_stem: str) -> str:
    prefix = f"{src_style}__"
    return src_stem[len(prefix) :] if src_stem.startswith(prefix) else src_stem


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def aggregate_metrics(
    *,
    label: str,
    source: str,
    step: int | None,
    metrics_csv: Path,
    images_dir: Path | None,
) -> EvalStats:
    rows = read_csv(metrics_csv)
    transfer = [row for row in rows if row["src_style"] != row["tgt_style"]]
    identity = [row for row in rows if row["src_style"] == row["tgt_style"]]

    def mean(items: list[dict[str, str]], key: str) -> float:
        return sum(metric_value(row, key) for row in items) / len(items)

    return EvalStats(
        label=label,
        source=source,
        step=step,
        metrics_csv=metrics_csv,
        images_dir=images_dir,
        count_all=len(rows),
        count_transfer=len(transfer),
        count_identity=len(identity),
        clip_all=mean(rows, "clip_style"),
        lpips_all=mean(rows, "lpips"),
        clip_transfer=mean(transfer, "clip_style"),
        lpips_transfer=mean(transfer, "lpips"),
        clip_identity=mean(identity, "clip_style"),
        lpips_identity=mean(identity, "lpips"),
    )


def load_new_curve() -> list[EvalStats]:
    curve: list[EvalStats] = []
    for eval_dir in sorted(NEW_SAMAM_ROOT.glob("eval_step_*")):
        step = int(eval_dir.name.split("_")[-1])
        step_dir = eval_dir / f"step_{step:06d}"
        metrics_csv = step_dir / "metrics.csv"
        images_dir = step_dir / "images"
        if not metrics_csv.exists():
            continue
        curve.append(
            aggregate_metrics(
                label=f"WikiArts5 step {step}",
                source="samam_wikiarts5_segmented",
                step=step,
                metrics_csv=metrics_csv,
                images_dir=images_dir,
            )
        )
    if not curve:
        raise FileNotFoundError(f"No eval steps found under {NEW_SAMAM_ROOT}")
    return curve


def build_lookup(metrics_csv: Path, images_dir: Path | None) -> dict[tuple[str, str, str], dict[str, str | Path]]:
    lookup: dict[tuple[str, str, str], dict[str, str | Path]] = {}
    for row in read_csv(metrics_csv):
        key = (
            row["src_style"],
            row["tgt_style"],
            canonical_src_stem(row["src_style"], row["src_stem"]),
        )
        payload = dict(row)
        if images_dir is not None:
            payload["images_dir"] = images_dir
        lookup[key] = payload
    return lookup


def resolve_source(src_style: str, src_stem: str) -> Path:
    direct = SOURCE_ROOT / src_style / f"{src_style}__{src_stem}.jpg"
    if direct.exists():
        return direct
    fallback = SOURCE_ROOT / src_style / f"{src_stem}.jpg"
    if fallback.exists():
        return fallback
    raise FileNotFoundError((src_style, src_stem))


def resolve_target_ref(tgt_style: str) -> Path:
    candidates = sorted((SOURCE_ROOT / tgt_style).glob("*.jpg"))
    if not candidates:
        raise FileNotFoundError(tgt_style)
    return candidates[0]


def resolve_generated_path(images_dir: Path, row: dict[str, str | Path]) -> Path:
    image_name = Path(str(row["image"])).name
    direct = images_dir / image_name
    if direct.exists():
        return direct
    raise FileNotFoundError(image_name)


def load_tile(path: Path) -> Image.Image:
    return Image.open(path).convert("RGB").resize((CELL - 4, CELL - 4), Image.Resampling.LANCZOS)


def build_case_image_map(
    case: dict[str, str],
    old_lookup: dict[tuple[str, str, str], dict[str, str | Path]],
    new_clip_lookup: dict[tuple[str, str, str], dict[str, str | Path]],
    new_lpips_lookup: dict[tuple[str, str, str], dict[str, str | Path]],
) -> dict[str, Image.Image]:
    key = (case["src_style"], case["tgt_style"], case["src_stem"])
    image_map: dict[str, Image.Image] = {
        "Source": load_tile(resolve_source(case["src_style"], case["src_stem"])),
        "IDT": load_tile(resolve_source(case["src_style"], case["src_stem"])),
        "SaMAM-2250": load_tile(resolve_generated_path(OLD_SAMAM_IMAGES, old_lookup[key])),
        "W5 best-CLIP": load_tile(resolve_generated_path(Path(str(new_clip_lookup[key]["images_dir"])), new_clip_lookup[key])),
        "W5 best-LPIPS": load_tile(resolve_generated_path(Path(str(new_lpips_lookup[key]["images_dir"])), new_lpips_lookup[key])),
        "Target ref": load_tile(resolve_target_ref(case["tgt_style"])),
    }
    return image_map


def build_case_metric_text(
    case: dict[str, str],
    idt_stats: EvalStats,
    old_lookup: dict[tuple[str, str, str], dict[str, str | Path]],
    new_clip_lookup: dict[tuple[str, str, str], dict[str, str | Path]],
    new_lpips_lookup: dict[tuple[str, str, str], dict[str, str | Path]],
) -> dict[str, str]:
    key = (case["src_style"], case["tgt_style"], case["src_stem"])
    old = old_lookup[key]
    new_clip = new_clip_lookup[key]
    new_lpips = new_lpips_lookup[key]
    return {
        "Source": "source",
        "IDT": f"clip {idt_stats.clip_transfer:.3f}",
        "SaMAM-2250": f"clip {metric_value(old, 'clip_style'):.3f} | lp {metric_value(old, 'lpips'):.3f}",
        "W5 best-CLIP": f"clip {metric_value(new_clip, 'clip_style'):.3f} | lp {metric_value(new_clip, 'lpips'):.3f}",
        "W5 best-LPIPS": f"clip {metric_value(new_lpips, 'clip_style'):.3f} | lp {metric_value(new_lpips, 'lpips'):.3f}",
        "Target ref": "ref",
    }


def annotate(ax, x: float, y: float, text: str, *, dx: float, dy: float, color: str) -> None:
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


def write_summary_files(
    *,
    idt_stats: EvalStats,
    old_samam: EvalStats,
    best_clip: EvalStats,
    best_lpips: EvalStats,
    latest: EvalStats,
    curve: list[EvalStats],
) -> None:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    rows = [
        {
            "label": "IDT",
            "source": idt_stats.source,
            "step": "",
            "transfer_clip_style": idt_stats.clip_transfer,
            "transfer_lpips": idt_stats.lpips_transfer,
            "all_pairs_clip_style": idt_stats.clip_all,
            "all_pairs_lpips": idt_stats.lpips_all,
            "identity_clip_style": idt_stats.clip_identity,
            "identity_lpips": idt_stats.lpips_identity,
            "delta_idt_transfer": 0.0,
        },
        {
            "label": "SaMAM-2250",
            "source": old_samam.source,
            "step": "",
            "transfer_clip_style": old_samam.clip_transfer,
            "transfer_lpips": old_samam.lpips_transfer,
            "all_pairs_clip_style": old_samam.clip_all,
            "all_pairs_lpips": old_samam.lpips_all,
            "identity_clip_style": old_samam.clip_identity,
            "identity_lpips": old_samam.lpips_identity,
            "delta_idt_transfer": old_samam.delta_idt(idt_stats.clip_transfer),
        },
        {
            "label": "WikiArts5 best-CLIP",
            "source": best_clip.source,
            "step": best_clip.step,
            "transfer_clip_style": best_clip.clip_transfer,
            "transfer_lpips": best_clip.lpips_transfer,
            "all_pairs_clip_style": best_clip.clip_all,
            "all_pairs_lpips": best_clip.lpips_all,
            "identity_clip_style": best_clip.clip_identity,
            "identity_lpips": best_clip.lpips_identity,
            "delta_idt_transfer": best_clip.delta_idt(idt_stats.clip_transfer),
        },
        {
            "label": "WikiArts5 best-LPIPS",
            "source": best_lpips.source,
            "step": best_lpips.step,
            "transfer_clip_style": best_lpips.clip_transfer,
            "transfer_lpips": best_lpips.lpips_transfer,
            "all_pairs_clip_style": best_lpips.clip_all,
            "all_pairs_lpips": best_lpips.lpips_all,
            "identity_clip_style": best_lpips.clip_identity,
            "identity_lpips": best_lpips.lpips_identity,
            "delta_idt_transfer": best_lpips.delta_idt(idt_stats.clip_transfer),
        },
        {
            "label": "WikiArts5 latest",
            "source": latest.source,
            "step": latest.step,
            "transfer_clip_style": latest.clip_transfer,
            "transfer_lpips": latest.lpips_transfer,
            "all_pairs_clip_style": latest.clip_all,
            "all_pairs_lpips": latest.lpips_all,
            "identity_clip_style": latest.clip_identity,
            "identity_lpips": latest.lpips_identity,
            "delta_idt_transfer": latest.delta_idt(idt_stats.clip_transfer),
        },
    ]
    with SUMMARY_CSV.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    curve_rows = [
        {
            "step": item.step,
            "transfer_clip_style": item.clip_transfer,
            "transfer_lpips": item.lpips_transfer,
            "all_pairs_clip_style": item.clip_all,
            "all_pairs_lpips": item.lpips_all,
            "identity_clip_style": item.clip_identity,
            "identity_lpips": item.lpips_identity,
            "delta_idt_transfer": item.delta_idt(idt_stats.clip_transfer),
            "delta_idt_all_pairs": item.clip_all - idt_stats.clip_all,
        }
        for item in curve
    ]
    with CURVE_CSV.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(curve_rows[0].keys()))
        writer.writeheader()
        writer.writerows(curve_rows)

    payload = {
        "fixed_test_split": str(SOURCE_ROOT),
        "idt": rows[0],
        "samam_2250": rows[1],
        "wikiarts5_best_clip": rows[2],
        "wikiarts5_best_lpips": rows[3],
        "wikiarts5_latest": rows[4],
        "curve_csv": str(CURVE_CSV),
    }
    SUMMARY_JSON.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def render_summary_figure(
    *,
    idt_stats: EvalStats,
    old_samam: EvalStats,
    best_clip: EvalStats,
    best_lpips: EvalStats,
    latest: EvalStats,
    curve: list[EvalStats],
) -> None:
    FIG_ROOT.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "DejaVu Serif"],
            "font.size": 9.0,
            "axes.labelsize": 9.2,
            "axes.titlesize": 10.0,
            "xtick.labelsize": 7.8,
            "ytick.labelsize": 7.8,
            "legend.fontsize": 7.4,
            "figure.dpi": 300,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.03,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.22,
            "grid.linewidth": 0.55,
        }
    )
    fig, axes = plt.subplots(1, 2, figsize=(7.7, 3.05), gridspec_kw={"width_ratios": [1.07, 1.0]})

    left = axes[0]
    left.set_facecolor("#FCFBF8")
    left.axhspan(-0.12, 0.0, color="#F2E8F7", alpha=0.30, zorder=0)
    left.axhline(0.0, color="#8E63C0", lw=1.5, ls=(0, (7, 4)), zorder=1)
    left.text(0.845, -0.014, "IDT floor", color="#8E63C0", fontsize=8.2, ha="right", weight="bold")

    xs = [item.one_minus_lpips_transfer for item in curve]
    ys = [item.delta_idt(idt_stats.clip_transfer) for item in curve]
    left.plot(xs, ys, color="#C7C7C7", lw=1.1, alpha=0.85, zorder=2)
    left.scatter(xs, ys, s=16, color="#BEBEBE", alpha=0.65, linewidths=0, zorder=2)

    points = [
        ("SaMAM-2250", old_samam, "#5D8FBF", "o", -22, -12),
        ("WikiArts5 best-CLIP", best_clip, "#D64045", "D", 8, 8),
        ("WikiArts5 best-LPIPS", best_lpips, "#B45309", "s", -106, -8),
        ("WikiArts5 latest", latest, "#3C8F89", "^", -70, 8),
    ]
    for label, stats, color, marker, dx, dy in points:
        x = stats.one_minus_lpips_transfer
        y = stats.delta_idt(idt_stats.clip_transfer)
        left.scatter([x], [y], s=78, c=color, marker=marker, edgecolors="white", linewidths=0.9, zorder=4)
        suffix = f" ({stats.step})" if stats.step is not None else ""
        annotate(left, x, y, f"{label}{suffix}", dx=dx, dy=dy, color=color)

    left.set_title("(a) Transfer frontier vs IDT on the fixed test split")
    left.set_xlabel(r"$1 - \mathrm{LPIPS}_{tr}$")
    left.set_ylabel(r"$\Delta_{\mathrm{IDT,tr}}$ (CLIP-S) $\uparrow$")
    left.set_xlim(0.62, 0.78)
    left.set_ylim(-0.11, 0.01)

    right = axes[1]
    steps = [item.step for item in curve if item.step is not None]
    clip_curve = [item.clip_transfer for item in curve]
    lpips_curve = [item.lpips_transfer for item in curve]
    right.set_facecolor("#FBFBFB")
    right.plot(steps, clip_curve, color="#D64045", lw=1.6, label="transfer CLIP-S")
    right.axhline(idt_stats.clip_transfer, color="#8E63C0", lw=1.3, ls=(0, (7, 4)), label="IDT transfer CLIP-S")
    right.scatter(
        [best_clip.step, best_lpips.step, latest.step],
        [best_clip.clip_transfer, best_lpips.clip_transfer, latest.clip_transfer],
        s=28,
        c=["#D64045", "#B45309", "#3C8F89"],
        edgecolors="white",
        linewidths=0.7,
        zorder=4,
    )
    right.set_xlabel("Training step")
    right.set_ylabel("transfer CLIP-S", color="#C21F34")
    right.tick_params(axis="y", labelcolor="#C21F34")
    right.set_title("(b) WikiArts-5 SaMAM trajectory")

    right2 = right.twinx()
    right2.plot(steps, lpips_curve, color="#1D4ED8", lw=1.4, label="transfer LPIPS")
    right2.scatter(
        [best_clip.step, best_lpips.step, latest.step],
        [best_clip.lpips_transfer, best_lpips.lpips_transfer, latest.lpips_transfer],
        s=28,
        c=["#D64045", "#B45309", "#3C8F89"],
        marker="s",
        edgecolors="white",
        linewidths=0.7,
        zorder=4,
    )
    right2.set_ylabel("transfer LPIPS", color="#1D4ED8")
    right2.tick_params(axis="y", labelcolor="#1D4ED8")
    right2.set_ylim(0.22, 0.62)

    handles1, labels1 = right.get_legend_handles_labels()
    handles2, labels2 = right2.get_legend_handles_labels()
    right.legend(handles1 + handles2, labels1 + labels2, loc="upper right")

    fig.savefig(SUMMARY_PNG)
    fig.savefig(SUMMARY_PDF)
    plt.close(fig)


def render_qualitative_figure(
    *,
    idt_stats: EvalStats,
    old_samam: EvalStats,
    best_clip: EvalStats,
    best_lpips: EvalStats,
) -> None:
    old_lookup = build_lookup(OLD_SAMAM_METRICS, OLD_SAMAM_IMAGES)
    new_clip_lookup = build_lookup(best_clip.metrics_csv, best_clip.images_dir)
    new_lpips_lookup = build_lookup(best_lpips.metrics_csv, best_lpips.images_dir)

    columns = ["Source", "IDT", "SaMAM-2250", "W5 best-CLIP", "W5 best-LPIPS", "Target ref"]
    cases = [FAIL_CASE, GAIN_CASE]
    width = LEFT_W + len(columns) * CELL
    height = TOP_PAD + TITLE_H + HEADER_H + len(cases) * (CELL + ROW_GAP)
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)

    title = "WikiArts-5 page1 diagnostic: larger train pool lifts SaMAM, but still stays under IDT."
    draw.text((8, TOP_PAD + 2), title, anchor="la", fill=(18, 18, 18), font=FONT_TITLE)
    y = TOP_PAD + TITLE_H
    for idx, col in enumerate(columns):
        draw.text((LEFT_W + idx * CELL + CELL // 2, y), col, anchor="ma", fill=(22, 22, 22), font=FONT_B)
    y += HEADER_H

    for case in cases:
        image_map = build_case_image_map(case, old_lookup, new_clip_lookup, new_lpips_lookup)
        text_map = build_case_metric_text(case, idt_stats, old_lookup, new_clip_lookup, new_lpips_lookup)
        draw.text((8, y + 34), case["row_label"], anchor="lm", fill=(28, 28, 28), font=FONT_ROW)
        for idx, col in enumerate(columns):
            x = LEFT_W + idx * CELL + 2
            canvas.paste(image_map[col], (x, y + 2))
            draw.rectangle([x, y + 2, x + CELL - 4, y + CELL - 2], outline=(188, 188, 188), width=1)
            draw.text((x + (CELL - 4) // 2, y + CELL - 2), text_map[col], anchor="ms", fill=(48, 48, 48), font=FONT_META)
        y += CELL + ROW_GAP

    meta = (
        f"IDT transfer CLIP-S = {idt_stats.clip_transfer:.4f}; "
        f"best-CLIP step = {best_clip.step}; best-LPIPS step = {best_lpips.step}; "
        f"all images come from the fixed Distinct5 test split."
    )
    draw.text((8, height - 10), meta, anchor="ls", fill=(74, 74, 74), font=FONT_META)

    FIG_ROOT.mkdir(parents=True, exist_ok=True)
    canvas.save(QUAL_PNG)
    canvas.save(QUAL_PDF, resolution=300.0)


def main() -> None:
    curve = load_new_curve()
    idt_stats = aggregate_metrics(
        label="IDT",
        source="distinct5_fixed_test",
        step=None,
        metrics_csv=IDT_METRICS,
        images_dir=None,
    )
    old_samam = aggregate_metrics(
        label="SaMAM-2250",
        source="samam_distinct5_old_train",
        step=None,
        metrics_csv=OLD_SAMAM_METRICS,
        images_dir=OLD_SAMAM_IMAGES,
    )
    best_clip = max(curve, key=lambda item: item.clip_transfer)
    best_lpips = min(curve, key=lambda item: item.lpips_transfer)
    latest = curve[-1]

    write_summary_files(
        idt_stats=idt_stats,
        old_samam=old_samam,
        best_clip=best_clip,
        best_lpips=best_lpips,
        latest=latest,
        curve=curve,
    )
    render_summary_figure(
        idt_stats=idt_stats,
        old_samam=old_samam,
        best_clip=best_clip,
        best_lpips=best_lpips,
        latest=latest,
        curve=curve,
    )
    render_qualitative_figure(
        idt_stats=idt_stats,
        old_samam=old_samam,
        best_clip=best_clip,
        best_lpips=best_lpips,
    )
    print(SUMMARY_JSON)
    print(SUMMARY_CSV)
    print(CURVE_CSV)
    print(SUMMARY_PNG)
    print(QUAL_PNG)


if __name__ == "__main__":
    main()
