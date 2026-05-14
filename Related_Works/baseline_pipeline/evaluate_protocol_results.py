from __future__ import annotations

import argparse
import csv
import importlib
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


PIPELINE_ROOT = Path(__file__).resolve().parent
WORKSPACE_ROOT = PIPELINE_ROOT.parent.parent
SB_ROOT = WORKSPACE_ROOT / "SchrodingerBridge"
SB_RUN_EVAL = SB_ROOT / "run_evaluation.py"
SB_SRC = SB_ROOT / "src"
STYLE_DATA = WORKSPACE_ROOT / "style_data"
OVERFIT50 = STYLE_DATA / "overfit50"
RESULTS_ROOT = PIPELINE_ROOT / "results"
STYLE_SUBDIRS = ["photo", "monet", "vangogh", "cezanne", "Hayao"]
DEFAULT_REFERENCE_IMAGES = (
    SB_ROOT
    / "exp"
    / "pareto_probe_4"
    / "S-add__K-3_C-2_W-10_Col-15"
    / "full_eval"
    / "epoch_0001"
    / "images"
)


def _reference_names(reference_images_dir: Path) -> set[str]:
    names = {p.name for p in reference_images_dir.iterdir() if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png"}}
    if not names:
        raise RuntimeError(f"No reference images found: {reference_images_dir}")
    return names


def _validate_images(images_dir: Path, reference_names: set[str]) -> tuple[int, list[str]]:
    found = {p.name for p in images_dir.iterdir() if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png"}}
    missing = sorted(reference_names - found)
    return len(found & reference_names), missing


def _summary_metrics(summary_path: Path) -> dict[str, Any]:
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    analysis = payload.get("analysis", {}) or {}
    all_pairs = analysis.get("all_pairs_overview", {}) or {}
    transfer = analysis.get("style_transfer_ability", {}) or {}
    photo = analysis.get("photo_to_art_performance", {}) or {}
    identity = analysis.get("identity_reconstruction", {}) or {}
    return {
        "clip_style": all_pairs.get("clip_style"),
        "clip_content": all_pairs.get("clip_content"),
        "content_lpips": all_pairs.get("content_lpips"),
        "fid": all_pairs.get("fid"),
        "art_fid": all_pairs.get("art_fid"),
        "cmmd": all_pairs.get("cmmd"),
        "dino_structure": all_pairs.get("dino_structure"),
        "gram_micro": all_pairs.get("gram_micro"),
        "gram_macro": all_pairs.get("gram_macro"),
        "transfer_clip_style": transfer.get("clip_style"),
        "transfer_clip_content": transfer.get("clip_content"),
        "transfer_content_lpips": transfer.get("content_lpips"),
        "photo_to_art_clip_style": photo.get("clip_style"),
        "photo_to_art_clip_content": photo.get("clip_content"),
        "photo_to_art_content_lpips": photo.get("content_lpips"),
        "identity_clip_content": identity.get("clip_content"),
        "identity_content_lpips": identity.get("content_lpips"),
    }


def _append_modern_metrics(eval_dir: Path, batch_size: int) -> str:
    if str(SB_SRC) not in sys.path:
        sys.path.insert(0, str(SB_SRC))
    modern_metrics = importlib.import_module("utils.modern_metrics")
    cfg = modern_metrics.ModernMetricConfig(
        test_dir=OVERFIT50,
        device="cuda" if shutil.which("nvidia-smi") else "cpu",
        clip_model_name="openai/clip-vit-base-patch32",
        dino_model_name="facebook/dinov2-small",
        cmmd_sigma=10.0,
        batch_size=batch_size,
    )
    modern_metrics.append_modern_metrics_to_summary(eval_dir, cfg)
    return "ok"


def _run_eval(result_root: Path, enable_artfid: bool) -> None:
    cmd = [
        sys.executable,
        str(SB_RUN_EVAL),
        f"--output={result_root}",
        f"--test_dir={OVERFIT50}",
        f"--style_subdirs={','.join(STYLE_SUBDIRS)}",
        "--reuse_generated",
        "--force_regen",
    ]
    if enable_artfid:
        cmd.append("--eval_enable_art_fid")
    else:
        cmd.append("--no-eval_enable_art_fid")
    subprocess.run(cmd, cwd=str(SB_ROOT), check=True)


def write_table(rows: list[dict[str, Any]], protocol: str) -> tuple[Path, Path]:
    out_json = RESULTS_ROOT / f"protocol_eval_table_{protocol}.json"
    out_csv = RESULTS_ROOT / f"protocol_eval_table_{protocol}.csv"
    out_json.write_text(json.dumps({"runs": rows}, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    fields = [
        "baseline",
        "protocol",
        "status",
        "images_matched",
        "images_expected",
        "eval_sec",
        "modern_sec",
        "error",
        "clip_style",
        "clip_content",
        "content_lpips",
        "fid",
        "art_fid",
        "cmmd",
        "dino_structure",
        "gram_micro",
        "gram_macro",
        "transfer_clip_style",
        "transfer_clip_content",
        "transfer_content_lpips",
        "photo_to_art_clip_style",
        "photo_to_art_clip_content",
        "photo_to_art_content_lpips",
        "identity_clip_content",
        "identity_content_lpips",
        "result_root",
    ]
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fields})
    return out_json, out_csv


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate standard protocol result folders and produce one comparison table.")
    parser.add_argument("--protocol", default="protocol_a_800")
    parser.add_argument("--baselines", nargs="+", required=True)
    parser.add_argument("--reference-images-dir", type=Path, default=DEFAULT_REFERENCE_IMAGES)
    parser.add_argument("--enable-artfid", action="store_true")
    parser.add_argument("--append-modern", action="store_true")
    parser.add_argument("--modern-batch-size", type=int, default=8)
    args = parser.parse_args()

    reference_names = _reference_names(args.reference_images_dir.resolve())
    rows: list[dict[str, Any]] = []
    for baseline in args.baselines:
        result_root = RESULTS_ROOT / baseline / args.protocol
        images_dir = result_root / "images"
        row: dict[str, Any] = {
            "baseline": baseline,
            "protocol": args.protocol,
            "result_root": str(result_root),
            "images_expected": len(reference_names),
            "eval_sec": None,
            "modern_sec": None,
            "error": "",
        }
        try:
            if not images_dir.is_dir():
                raise FileNotFoundError(f"missing images dir: {images_dir}")
            matched, missing = _validate_images(images_dir, reference_names)
            row["images_matched"] = matched
            if missing:
                raise RuntimeError(f"missing {len(missing)} reference images, examples: {', '.join(missing[:10])}")

            t0 = time.time()
            _run_eval(result_root, enable_artfid=bool(args.enable_artfid))
            row["eval_sec"] = round(time.time() - t0, 3)

            if args.append_modern:
                t1 = time.time()
                _append_modern_metrics(result_root, batch_size=int(args.modern_batch_size))
                row["modern_sec"] = round(time.time() - t1, 3)

            summary_path = result_root / "summary.json"
            row.update(_summary_metrics(summary_path))
            row["status"] = "ok"
        except Exception as exc:
            row["status"] = "failed"
            row["error"] = f"{type(exc).__name__}: {exc}"
        rows.append(row)

    json_path, csv_path = write_table(rows, args.protocol)
    print(f"table json: {json_path}")
    print(f"table csv : {csv_path}")
    for row in rows:
        print(f"{row['baseline']}: {row['status']} matched={row.get('images_matched')} error={row.get('error')}")
    return 1 if any(row.get("status") != "ok" for row in rows) else 0


if __name__ == "__main__":
    raise SystemExit(main())
