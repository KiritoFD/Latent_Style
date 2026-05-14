"""Collect Related_Works reproduction data into CSV and Markdown indexes.

This script is intentionally read-only with respect to experiment outputs. It
only scans existing files and writes lightweight index/report files.
"""
from __future__ import annotations

import csv
import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
RELATED_ROOT = SCRIPT_DIR.parent
WORKSPACE_ROOT = RELATED_ROOT.parent
RUN511_ROOT = RELATED_ROOT / "run_511"
COMPLETE_ROOT = RUN511_ROOT / "complete_750"
OUTPUTS_ROOT = RUN511_ROOT / "outputs"
LEGACY_RUNS_ROOT = RELATED_ROOT / "runs"
RESULTS_DIR = RELATED_ROOT / "results"
DOCS_DIR = RELATED_ROOT / "docs"
METRICS_DIR = RESULTS_DIR / "metrics_summary"
JSON_ARCHIVE_DIR = RESULTS_DIR / "json_archive"

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
DATA_EXTS = {".csv", ".json", ".md", ".html"}


def rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(WORKSPACE_ROOT.resolve())).replace("\\", "/")
    except Exception:
        return str(path).replace("\\", "/")


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def all_row(path: Path) -> dict[str, Any]:
    data = read_json(path)
    for row in data.get("results", []):
        if str(row.get("target", "")).upper() == "ALL":
            return row
    return {}


def image_count(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(1 for p in path.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS)


def count_data_files(path: Path, suffix: str) -> int:
    if not path.exists():
        return 0
    return sum(1 for p in path.rglob(f"*{suffix}") if p.is_file())


def first_summary_times(path: Path) -> dict[str, Any]:
    summary = read_json(path / "summary.json")
    result: dict[str, Any] = {
        "has_summary": bool(summary),
        "train_status": "",
        "train_sec": "",
        "infer_status": "",
        "infer_sec": "",
        "sec_per_image": "",
    }
    for run in summary.get("runs", []):
        stage = run.get("stage")
        if stage == "train" and result["train_sec"] == "":
            result["train_status"] = run.get("status", "")
            result["train_sec"] = run.get("elapsed_sec", "")
        if stage == "infer" and result["infer_sec"] == "":
            result["infer_status"] = run.get("status", "")
            result["infer_sec"] = run.get("elapsed_sec", "")
            result["sec_per_image"] = run.get("sec_per_image", "")
    return result


def fmt_num(value: Any, digits: int = 4) -> str:
    if value in ("", None):
        return ""
    try:
        return f"{float(value):.{digits}f}"
    except Exception:
        return str(value)


def build_complete_rows() -> list[dict[str, Any]]:
    manifest = read_json(COMPLETE_ROOT / "manifest.json")
    rows: list[dict[str, Any]] = []
    for item in manifest.get("runs", []):
        run = item.get("run", "")
        run_dir = COMPLETE_ROOT / run
        protocol = all_row(run_dir / "eval_protocol750_sbmatch.json")
        guard = all_row(run_dir / "eval_guard750.json")
        artifact = all_row(run_dir / "eval_artifact_pack750.json")
        hf_kid = all_row(run_dir / "eval_hf_patch_kid750.json")
        plain_kid = all_row(run_dir / "eval_plain_kid750.json")
        rows.append(
            {
                "source": "complete_750",
                "run": run,
                "path": rel(run_dir),
                "images": item.get("images", image_count(run_dir / "images")),
                "ref_match": item.get("ref_match", ""),
                "lpips": protocol.get("lpips", ""),
                "clip_style": protocol.get("clip_style", ""),
                "clip_content": protocol.get("clip_content", ""),
                "ssim_y": guard.get("ssim_y", ""),
                "edge_f1": guard.get("edge_f1", ""),
                "blur_style_drop": guard.get("blur_style_drop", ""),
                "down_style_drop": guard.get("down_style_drop", ""),
                "extra_edge_rate": guard.get("extra_edge_rate", ""),
                "chroma_speckle_z": guard.get("chroma_speckle_z", ""),
                "flat_chroma_hf_z": guard.get("flat_chroma_hf_z", ""),
                "musiq": artifact.get("musiq", ""),
                "maniqa": artifact.get("maniqa", ""),
                "dists_content": artifact.get("dists_content", ""),
                "fft_radial_kl_style": artifact.get("fft_radial_kl_style", ""),
                "fft_slope_error": artifact.get("fft_slope_error", ""),
                "chroma_grain_index": artifact.get("chroma_grain_index", ""),
                "hf_patch_kid": hf_kid.get("hf_patch_kid", ""),
                "plain_kid": plain_kid.get("kid", ""),
                "has_protocol_eval": bool(protocol),
                "has_guard_eval": bool(guard),
                "has_artifact_pack": bool(artifact),
                "has_hf_patch_kid": bool(hf_kid),
                "has_plain_kid": bool(plain_kid),
            }
        )
    return rows


def build_output_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not OUTPUTS_ROOT.exists():
        return rows
    for run_dir in sorted(p for p in OUTPUTS_ROOT.iterdir() if p.is_dir()):
        infer_dir = run_dir / "infer_750"
        protocol = all_row(infer_dir / "eval_protocol750_sbmatch.json")
        guard = all_row(infer_dir / "eval_guard750.json")
        times = first_summary_times(run_dir)
        rows.append(
            {
                "source": "run_511_outputs",
                "run": run_dir.name,
                "path": rel(run_dir),
                "images": image_count(infer_dir / "images"),
                "checkpoints": count_data_files(run_dir / "checkpoints", ".pth")
                + count_data_files(run_dir / "checkpoints", ".pt")
                + count_data_files(run_dir / "checkpoints", ".model")
                + count_data_files(run_dir / "checkpoints", ".ckpt"),
                "has_summary": times["has_summary"],
                "train_status": times["train_status"],
                "train_sec": times["train_sec"],
                "infer_status": times["infer_status"],
                "infer_sec": times["infer_sec"],
                "sec_per_image": times["sec_per_image"],
                "lpips": protocol.get("lpips", ""),
                "clip_style": protocol.get("clip_style", ""),
                "clip_content": protocol.get("clip_content", ""),
                "ssim_y": guard.get("ssim_y", ""),
                "edge_f1": guard.get("edge_f1", ""),
                "has_protocol_eval": bool(protocol),
                "has_guard_eval": bool(guard),
            }
        )
    return rows


def build_legacy_image_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not LEGACY_RUNS_ROOT.exists():
        return rows
    for img_dir in sorted(p for p in LEGACY_RUNS_ROOT.rglob("*") if p.is_dir() and p.name.lower() == "images"):
        count = image_count(img_dir)
        if count == 0:
            continue
        rows.append(
            {
                "source": "legacy_runs_images",
                "run": img_dir.parent.name,
                "path": rel(img_dir),
                "images": count,
                "parent": rel(img_dir.parent),
            }
        )
    return rows


def build_data_file_rows() -> list[dict[str, Any]]:
    roots = [
        RELATED_ROOT / "summary",
        RELATED_ROOT / "results",
        RELATED_ROOT / "docs",
        RUN511_ROOT / "docs",
        COMPLETE_ROOT,
    ]
    rows: list[dict[str, Any]] = []
    seen: set[Path] = set()
    for root in roots:
        if not root.exists():
            continue
        for path in sorted(p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in DATA_EXTS):
            resolved = path.resolve()
            if resolved in seen:
                continue
            seen.add(resolved)
            rows.append(
                {
                    "category": rel(root),
                    "path": rel(path),
                    "suffix": path.suffix.lower(),
                    "bytes": path.stat().st_size,
                    "modified": datetime.fromtimestamp(path.stat().st_mtime).isoformat(timespec="seconds"),
                }
            )
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def write_all_metrics_csv(rows: list[dict[str, Any]]) -> None:
    out = COMPLETE_ROOT / "summary_all_tested_metrics.csv"
    write_csv(out, rows)
    write_csv(METRICS_DIR / "protocol750_all_tested_metrics.csv", rows)


def copy_metric_summary_files() -> None:
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    for src in sorted(COMPLETE_ROOT.glob("summary*.csv")) + sorted(COMPLETE_ROOT.glob("summary*.md")):
        if src.name == "manifest.json":
            continue
        shutil.copy2(src, METRICS_DIR / src.name)
    for src in [
        RUN511_ROOT / "docs" / "timing_summary.csv",
        RUN511_ROOT / "docs" / "timing_summary.md",
        RUN511_ROOT / "docs" / "timing_filled_report.md",
        RUN511_ROOT / "docs" / "outputs_inventory.csv",
        RUN511_ROOT / "docs" / "outputs_inventory.md",
        RUN511_ROOT / "docs" / "protocol750_eval_report.csv",
        RUN511_ROOT / "docs" / "protocol750_eval_report.md",
    ]:
        if src.exists():
            shutil.copy2(src, METRICS_DIR / src.name)


def write_markdown(
    inventory_rows: list[dict[str, Any]],
    complete_rows: list[dict[str, Any]],
    output_rows: list[dict[str, Any]],
    legacy_rows: list[dict[str, Any]],
    data_rows: list[dict[str, Any]],
) -> None:
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    md = [
        "# Reproduction Data Index",
        "",
        f"Updated: {datetime.now().isoformat(timespec='seconds')}",
        "",
        "This is the current ledger for Related_Works reproduction data. It is generated by `Related_Works/scripts/collect_repro_inventory.py`.",
        "",
        "## Canonical CSV Files",
        "",
        "- `Related_Works/results/repro_data_inventory.csv`: run/output inventory with image counts, timing fields, and available metric coverage.",
        "- `Related_Works/results/repro_data_files.csv`: CSV/JSON/Markdown/HTML data-file ledger under `summary`, `results`, `docs`, `run_511/docs`, and `run_511/complete_750`.",
        "- `Related_Works/run_511/complete_750/summary_all_tested_metrics.csv`: wide metric table for all strict protocol-750 runs.",
        "- `Related_Works/results/metrics_summary/`: standalone metric-summary folder for paper/result aggregation.",
        "- `Related_Works/results/json_archive/`: archived top-level aggregate JSON files.",
        "",
        "## Strict Protocol-750 Results",
        "",
        "| Run | Images | LPIPS | CLIP-style | CLIP-content | SSIM-Y | Edge-F1 | MUSIQ | HF-Patch-KID | plain KID | Coverage |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in complete_rows:
        coverage = ",".join(
            name
            for name, ok in [
                ("protocol", row.get("has_protocol_eval")),
                ("guard", row.get("has_guard_eval")),
                ("artifact", row.get("has_artifact_pack")),
                ("hf-kid", row.get("has_hf_patch_kid")),
                ("kid", row.get("has_plain_kid")),
            ]
            if ok
        )
        md.append(
            f"| `{row['run']}` | {row.get('images', '')} | {fmt_num(row.get('lpips'))} | "
            f"{fmt_num(row.get('clip_style'))} | {fmt_num(row.get('clip_content'))} | "
            f"{fmt_num(row.get('ssim_y'))} | {fmt_num(row.get('edge_f1'))} | "
            f"{fmt_num(row.get('musiq'))} | {fmt_num(row.get('hf_patch_kid'), 6)} | "
            f"{fmt_num(row.get('plain_kid'), 6)} | {coverage} |"
        )
    md.extend(
        [
            "",
            "## run_511 Output Status",
            "",
            "| Run | Images | Ckpts | Train | Train sec | Infer | Infer sec | LPIPS | CLIP-style |",
            "| --- | ---: | ---: | --- | ---: | --- | ---: | ---: | ---: |",
        ]
    )
    for row in output_rows:
        md.append(
            f"| `{row['run']}` | {row.get('images', '')} | {row.get('checkpoints', '')} | "
            f"{row.get('train_status', '')} | {fmt_num(row.get('train_sec'))} | "
            f"{row.get('infer_status', '')} | {fmt_num(row.get('infer_sec'))} | "
            f"{fmt_num(row.get('lpips'))} | {fmt_num(row.get('clip_style'))} |"
        )
    md.extend(
        [
            "",
            "## Legacy Reusable Image Folders",
            "",
            "| Folder | Images | Parent |",
            "| --- | ---: | --- |",
        ]
    )
    for row in legacy_rows[:80]:
        md.append(f"| `{row['path']}` | {row.get('images', '')} | `{row.get('parent', '')}` |")
    if len(legacy_rows) > 80:
        md.append(f"| ... | ... | {len(legacy_rows) - 80} more rows in CSV |")
    md.extend(
        [
            "",
            "## Data File Ledger",
            "",
            f"- Data files indexed: `{len(data_rows)}`",
            f"- Inventory rows indexed: `{len(inventory_rows)}`",
            "",
            "Use `Related_Works/results/repro_data_files.csv` for the full file-level list.",
        ]
    )
    (DOCS_DIR / "REPRO_DATA_INDEX.md").write_text("\n".join(md) + "\n", encoding="utf-8")


def main() -> int:
    complete_rows = build_complete_rows()
    output_rows = build_output_rows()
    legacy_rows = build_legacy_image_rows()
    data_rows = build_data_file_rows()
    inventory_rows = complete_rows + output_rows + legacy_rows

    write_csv(RESULTS_DIR / "repro_data_inventory.csv", inventory_rows)
    write_csv(RESULTS_DIR / "repro_data_files.csv", data_rows)
    write_all_metrics_csv(complete_rows)
    copy_metric_summary_files()
    write_markdown(inventory_rows, complete_rows, output_rows, legacy_rows, data_rows)

    print(RESULTS_DIR / "repro_data_inventory.csv")
    print(RESULTS_DIR / "repro_data_files.csv")
    print(DOCS_DIR / "REPRO_DATA_INDEX.md")
    print(COMPLETE_ROOT / "summary_all_tested_metrics.csv")
    print(METRICS_DIR)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
