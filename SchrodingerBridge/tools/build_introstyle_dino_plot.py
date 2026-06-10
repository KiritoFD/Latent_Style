from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import torch

from utils.modern_metrics import DinoStructureEmbedder


ROOT = Path(__file__).resolve().parents[1]
AAAI_DIR = ROOT / "aaai2027"
PAGE1_SUMMARY = AAAI_DIR / "introstyle_page1" / "introstyle_page1_summary.csv"
INTROSTYLE_SUMMARY = AAAI_DIR / "current_introstyle_summary_20260609.csv"
OUT_CSV = AAAI_DIR / "current_introstyle_dino_summary_20260609.csv"
OUT_PNG = AAAI_DIR / "current_introstyle_vs_dino_20260609.png"
TEST_DIR = Path(r"F:\wikiart_distinct5_samam_512_classview\test")
BODYDECODER_LOCAL_ROOT = AAAI_DIR / "bodydecoder_introstyle_clean_v3_mirror"


BODYDECODER_ROWS = [
    {
        "label": "Hold4Mid bodydecoder e8",
        "run": "aaai2027_inmortal_hold4mid_e8_spatial_carriergate_bodydecoder_seed42_b8a2/epoch_0008",
        "images_dir": BODYDECODER_LOCAL_ROOT / "epoch_0008" / "images",
        "metrics_csv": BODYDECODER_LOCAL_ROOT / "epoch_0008" / "metrics.csv",
    },
    {
        "label": "Hold4Mid bodydecoder e12",
        "run": "aaai2027_inmortal_hold4mid_e8_spatial_carriergate_bodydecoder_seed42_b8a2/epoch_0012",
        "images_dir": BODYDECODER_LOCAL_ROOT / "epoch_0012" / "images",
        "metrics_csv": BODYDECODER_LOCAL_ROOT / "epoch_0012" / "metrics.csv",
    },
]


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _resolve_gen_path(images_dir: Path, raw: str) -> Path:
    name = Path(str(raw)).name
    direct = images_dir / name
    if direct.exists():
        return direct
    nested = images_dir / str(raw)
    if nested.exists():
        return nested
    return direct


def _dino_for_metrics(images_dir: Path, metrics_csv: Path, *, embedder: DinoStructureEmbedder, batch_size: int) -> tuple[int, float | None]:
    rows = _read_csv(metrics_csv)
    src_paths: list[Path] = []
    gen_paths: list[Path] = []
    for row in rows:
        src = TEST_DIR / row["src_style"] / row["src_image"]
        gen_name = row.get("gen_image") or row.get("image") or ""
        gen = _resolve_gen_path(images_dir, gen_name)
        if src.exists() and gen.exists():
            src_paths.append(src)
            gen_paths.append(gen)
    if not src_paths:
        return 0, None
    return len(src_paths), embedder.structure_distance(src_paths, gen_paths, batch_size=batch_size)


def _load_points() -> list[dict[str, object]]:
    intro_rows = _read_csv(INTROSTYLE_SUMMARY)
    page1_lookup = {row["run"]: row for row in _read_csv(PAGE1_SUMMARY)}
    points: list[dict[str, object]] = []
    for row in intro_rows:
        if row["status"] == "pending":
            continue
        run = row["run"]
        label = row["label"]
        if run in page1_lookup:
            page1 = page1_lookup[run]
            points.append(
                {
                    "label": label,
                    "run": run,
                    "status": row["status"],
                    "introstyle_delta_idt": float(row["introstyle_delta_idt"]),
                    "introstyle_target_score": float(row["introstyle_target_score"]),
                    "transfer_content_lpips": float(row["transfer_content_lpips"]),
                    "images_dir": Path(page1["images_dir"]),
                    "metrics_csv": Path(page1["metrics_csv"]),
                }
            )
            continue
        for body_row in BODYDECODER_ROWS:
            if body_row["run"] == run:
                points.append(
                    {
                        "label": label,
                        "run": run,
                        "status": row["status"],
                        "introstyle_delta_idt": float(row["introstyle_delta_idt"]),
                        "introstyle_target_score": float(row["introstyle_target_score"]),
                        "transfer_content_lpips": float(row["transfer_content_lpips"]),
                        "images_dir": body_row["images_dir"],
                        "metrics_csv": body_row["metrics_csv"],
                    }
                )
                break
    return points


def _write_csv(rows: list[dict[str, object]]) -> None:
    fieldnames = [
        "label",
        "run",
        "status",
        "n_pairs",
        "dino_structure",
        "introstyle_delta_idt",
        "introstyle_target_score",
        "transfer_content_lpips",
        "images_dir",
        "metrics_csv",
    ]
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with OUT_CSV.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _plot(rows: list[dict[str, object]]) -> None:
    colors = {
        "IDT": "#888888",
        "Seedream-4.5": "#d94841",
        "LBM-Knee": "#1f77b4",
        "LBM-PS-v2": "#ff7f0e",
        "Hold4Mid bodydecoder e8": "#2ca02c",
        "Hold4Mid bodydecoder e12": "#188a53",
    }
    fig, ax = plt.subplots(figsize=(8.6, 6.2), dpi=170)
    for row in rows:
        x = float(row["dino_structure"])
        y = float(row["introstyle_delta_idt"])
        label = str(row["label"])
        color = colors.get(label, "#555555")
        size = 92 if label in colors else 62
        ax.scatter(x, y, s=size, color=color, edgecolor="white", linewidth=0.8, zorder=3)
        ax.text(x + 0.0012, y + 0.0006, label, fontsize=8, color=color if label in colors else "#333333")
    ax.axhline(0.0, color="#999999", linestyle="--", linewidth=1.0, zorder=1)
    ax.grid(True, alpha=0.22, zorder=0)
    ax.invert_xaxis()
    ax.set_xlabel("DINO structure distance (lower is better, so right is worse)")
    ax.set_ylabel("IntroStyle delta-IDT")
    ax.set_title("Distinct5 Current IntroStyle vs DINO Structure")
    ax.text(0.985, 0.03, "Higher style is better ↑\nLower DINO distance is better ←", transform=ax.transAxes, ha="right", va="bottom", fontsize=8, color="#555555")
    fig.tight_layout()
    fig.savefig(OUT_PNG, bbox_inches="tight")


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    embedder = DinoStructureEmbedder("facebook/dinov2-small", device)
    points = _load_points()
    out_rows: list[dict[str, object]] = []
    for point in points:
        n_pairs, dino = _dino_for_metrics(Path(point["images_dir"]), Path(point["metrics_csv"]), embedder=embedder, batch_size=8)
        out_rows.append(
            {
                "label": point["label"],
                "run": point["run"],
                "status": point["status"],
                "n_pairs": n_pairs,
                "dino_structure": dino,
                "introstyle_delta_idt": point["introstyle_delta_idt"],
                "introstyle_target_score": point["introstyle_target_score"],
                "transfer_content_lpips": point["transfer_content_lpips"],
                "images_dir": str(point["images_dir"]),
                "metrics_csv": str(point["metrics_csv"]),
            }
        )
    _write_csv(out_rows)
    _plot([row for row in out_rows if row["dino_structure"] is not None])
    print(OUT_CSV)
    print(OUT_PNG)


if __name__ == "__main__":
    main()
