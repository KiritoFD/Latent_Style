from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
WORKSPACE = ROOT.parent.parent
OUT_DIR = ROOT / "introstyle_page1"

POINTS = [
    {
        "method": "IDT",
        "run": "IDT",
        "images_dir": str(WORKSPACE / "SchrodingerBridge" / "docs" / "experiments" / "idt_eval_20260602" / "distinct5_512" / "idt_5x5" / "images"),
        "metrics_csv": str(WORKSPACE / "SchrodingerBridge" / "docs" / "experiments" / "idt_eval_20260602" / "distinct5_512" / "idt_5x5" / "metrics.csv"),
        "plot_label": "IDT",
    },
    {
        "method": "SaMAM",
        "run": "SaMAM_2250",
        "images_dir": str(OUT_DIR / "staging" / "SaMAM_2250" / "images"),
        "metrics_csv": str(OUT_DIR / "staging" / "SaMAM_2250" / "metrics.csv"),
        "plot_label": "SaMAM-2250",
    },
    {
        "method": "SaMST",
        "run": "SaMST_e15",
        "images_dir": str(OUT_DIR / "staging" / "SaMST_e15" / "images"),
        "metrics_csv": str(OUT_DIR / "staging" / "SaMST_e15" / "metrics.csv"),
        "plot_label": "SaMST e15",
    },
    {
        "method": "SaMAM-latent",
        "run": "Lat_SaMAM_step1500",
        "images_dir": str(OUT_DIR / "staging" / "Lat_SaMAM_step1500" / "images"),
        "metrics_csv": str(OUT_DIR / "staging" / "Lat_SaMAM_step1500" / "metrics.csv"),
        "plot_label": "Lat SaMAM",
    },
    {
        "method": "SaMST-latent",
        "run": "Lat_SaMST_batch1050",
        "images_dir": str(OUT_DIR / "staging" / "Lat_SaMST_batch1050" / "images"),
        "metrics_csv": str(OUT_DIR / "staging" / "Lat_SaMST_batch1050" / "metrics.csv"),
        "plot_label": "Lat SaMST",
    },
    {
        "method": "LBM",
        "run": "LBM-K_e1",
        "images_dir": str(WORKSPACE / "SchrodingerBridge" / "exp" / "distinct5_512_ema_variant_k_content_adaptive_vq_queue_e3_b44_remote" / "full_eval" / "epoch_0001" / "images"),
        "metrics_csv": str(WORKSPACE / "SchrodingerBridge" / "exp" / "distinct5_512_ema_variant_k_content_adaptive_vq_queue_e3_b44_remote" / "full_eval" / "epoch_0001" / "metrics.csv"),
        "plot_label": "LBM-K",
    },
    {
        "method": "LBM",
        "run": "LBM-Knee_e13",
        "images_dir": str(OUT_DIR / "staging" / "LBM-Knee_e13" / "images"),
        "metrics_csv": str(OUT_DIR / "staging" / "LBM-Knee_e13" / "metrics.csv"),
        "plot_label": "LBM-Knee",
    },
    {
        "method": "LBM",
        "run": "LBM-PS-v2_e13",
        "images_dir": str(OUT_DIR / "staging" / "LBM-PS-v2_e13" / "images"),
        "metrics_csv": str(OUT_DIR / "staging" / "LBM-PS-v2_e13" / "metrics.csv"),
        "plot_label": "LBM-PS-v2",
    },
    {
        "method": "Seedream",
        "run": "Seedream_repaired750",
        "images_dir": str(OUT_DIR / "staging" / "Seedream_repaired750" / "images"),
        "metrics_csv": str(OUT_DIR / "staging" / "Seedream_repaired750" / "metrics.csv"),
        "plot_label": "Seedream-4.5",
    },
]


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write_rows(path: Path, rows: list[dict[str, str]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def materialize_smoke_metrics(*, sample_rows: int) -> list[dict[str, str]]:
    metrics_dir = OUT_DIR / "smoke_metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    manifest_rows: list[dict[str, str]] = []
    for point in POINTS:
        src = Path(point["metrics_csv"])
        rows = read_rows(src)
        keep = rows[:sample_rows]
        out_path = metrics_dir / f"{point['run']}_smoke{sample_rows}_metrics.csv"
        fieldnames = list(keep[0].keys()) if keep else (list(rows[0].keys()) if rows else [])
        write_rows(out_path, keep, fieldnames)
        manifest_rows.append(
            {
                "method": str(point["method"]),
                "run": str(point["run"]),
                "images_dir": str(point["images_dir"]),
                "metrics_csv": str(out_path.as_posix()),
                "plot_label": str(point["plot_label"]),
            }
        )
    return manifest_rows


def build_summary(probe_rows: list[dict[str, str]], *, manifest_rows: list[dict[str, str]]) -> list[dict[str, object]]:
    manifest_by_run = {row["run"]: row for row in manifest_rows}
    idt_row = next(row for row in probe_rows if row["run"] == "IDT")
    idt_target = float(idt_row["transfer_target_style_score"])
    out: list[dict[str, object]] = []
    for row in probe_rows:
        run = str(row["run"])
        manifest = manifest_by_run.get(run, {})
        target = float(row["transfer_target_style_score"])
        margin = float(row["transfer_style_margin"])
        out.append(
            {
                "method": row["method"],
                "run": run,
                "plot_label": manifest.get("plot_label", run),
                "images": int(row["images"]),
                "transfer_target_style_score": target,
                "transfer_source_style_score": float(row["transfer_source_style_score"]),
                "transfer_best_non_target_score": float(row["transfer_best_non_target_score"]),
                "transfer_style_margin": margin,
                "identity_target_style_score": float(row["identity_target_style_score"]),
                "transfer_delta_idt_style": target - idt_target,
                "images_dir": row["images_dir"],
                "metrics_csv": row["metrics_csv"],
            }
        )
    return out


def write_summary_markdown(rows: list[dict[str, object]], path: Path) -> None:
    lines = [
        "# IntroStyle Page-1 Shortlist",
        "",
        "| Label | Method | Transfer target | Delta-IDT | Style margin | Identity target |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['plot_label']} | {row['method']} | "
            f"{float(row['transfer_target_style_score']):.4f} | "
            f"{float(row['transfer_delta_idt_style']):.4f} | "
            f"{float(row['transfer_style_margin']):.4f} | "
            f"{float(row['identity_target_style_score']):.4f} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample-rows", type=int, default=20)
    parser.add_argument("--probe-results-csv", type=Path, default=OUT_DIR / "introstyle_page1_probe.csv")
    args = parser.parse_args()

    manifest_rows = materialize_smoke_metrics(sample_rows=int(args.sample_rows))
    manifest_path = OUT_DIR / "introstyle_page1_manifest.csv"
    write_rows(
        manifest_path,
        manifest_rows,
        ["method", "run", "images_dir", "metrics_csv", "plot_label"],
    )

    outputs = {
        "manifest_csv": str(manifest_path),
    }
    if args.probe_results_csv.exists():
        probe_rows = read_rows(args.probe_results_csv)
        summary_rows = build_summary(probe_rows, manifest_rows=manifest_rows)
        summary_csv = OUT_DIR / "introstyle_page1_summary.csv"
        summary_json = OUT_DIR / "introstyle_page1_summary.json"
        summary_md = OUT_DIR / "introstyle_page1_summary.md"
        write_rows(
            summary_csv,
            [{k: row[k] for k in [
                "method",
                "run",
                "plot_label",
                "images",
                "transfer_target_style_score",
                "transfer_source_style_score",
                "transfer_best_non_target_score",
                "transfer_style_margin",
                "identity_target_style_score",
                "transfer_delta_idt_style",
                "images_dir",
                "metrics_csv",
            ]} for row in summary_rows],
            [
                "method",
                "run",
                "plot_label",
                "images",
                "transfer_target_style_score",
                "transfer_source_style_score",
                "transfer_best_non_target_score",
                "transfer_style_margin",
                "identity_target_style_score",
                "transfer_delta_idt_style",
                "images_dir",
                "metrics_csv",
            ],
        )
        summary_json.write_text(json.dumps(summary_rows, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        write_summary_markdown(summary_rows, summary_md)
        outputs.update(
            {
                "summary_csv": str(summary_csv),
                "summary_json": str(summary_json),
                "summary_md": str(summary_md),
            }
        )

    print(json.dumps(outputs, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
