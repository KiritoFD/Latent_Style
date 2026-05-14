from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
DEFAULT_WRITE_ROOT = ROOT.parent / "review_additional_experiments_aggregates"
DEFAULT_CANDIDATES = [
    ROOT / "review_additional_experiments" / "review_additional_experiments",
    ROOT / "review_additional_experiments",
]


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _pick_output_root() -> Path:
    for candidate in DEFAULT_CANDIDATES:
        if (candidate / "step_count_sweep" / "status.csv").exists() or (candidate / "lambda_grid" / "status.csv").exists():
            return candidate
    raise SystemExit("No review_additional_experiments output root with step/lambda status found.")


def _safe_float(value: Any) -> float | None:
    try:
        if value in ("", None):
            return None
        return float(value)
    except Exception:
        return None


def _ec(style: float | None, lpips: float | None) -> float | None:
    if style is None or lpips is None:
        return None
    return float(style * (1.0 - lpips))


def _ensure_efficiency_profile(output_root: Path, write_root: Path) -> Path:
    out_path = write_root / "efficiency_profile.json"
    if out_path.exists():
        return out_path
    import run_review_experiments as rre

    rre._profile_checkpoint(
        ckpt_path=rre.DEFAULT_BASE_CKPT,
        output_path=out_path,
        step_values=[1, 4, 8, 12, 16],
        batch_sizes=[1, 4],
        warmup_iters=10,
        measure_iters=50,
        step_size=1.0,
        style_strength=None,
        residual_scale=1.0,
        dry_run=False,
    )
    return out_path


def summarize_step_sweep(output_root: Path) -> list[dict[str, Any]]:
    status_rows = _read_csv(output_root / "step_count_sweep" / "status.csv")
    rows: list[dict[str, Any]] = []
    for status_row in status_rows:
        num_steps = int(status_row["num_steps"])
        summary_path = output_root / "step_count_sweep" / f"steps_{num_steps:02d}" / "summary.json"
        summary = _load_json(summary_path)
        all_pairs = summary.get("analysis", {}).get("all_pairs_overview", {})
        transfer = summary.get("analysis", {}).get("style_transfer_ability", {})
        photo_to_art = summary.get("analysis", {}).get("photo_to_art_performance", {})
        elapsed = _safe_float(status_row.get("elapsed_sec"))
        rows.append(
            {
                "num_steps": num_steps,
                "eval_elapsed_sec": elapsed,
                "end_to_end_img_per_sec": (750.0 / elapsed) if elapsed else None,
                "all_clip_style": _safe_float(all_pairs.get("clip_style")),
                "all_clip_content": _safe_float(all_pairs.get("clip_content")),
                "all_content_lpips": _safe_float(all_pairs.get("content_lpips")),
                "all_ec": _ec(_safe_float(all_pairs.get("clip_style")), _safe_float(all_pairs.get("content_lpips"))),
                "transfer_clip_style": _safe_float(transfer.get("clip_style")),
                "transfer_clip_content": _safe_float(transfer.get("clip_content")),
                "transfer_content_lpips": _safe_float(transfer.get("content_lpips")),
                "transfer_ec": _ec(_safe_float(transfer.get("clip_style")), _safe_float(transfer.get("content_lpips"))),
                "photo_to_art_clip_style": _safe_float(photo_to_art.get("clip_style")),
                "photo_to_art_clip_content": _safe_float(photo_to_art.get("clip_content")),
                "photo_to_art_content_lpips": _safe_float(photo_to_art.get("content_lpips")),
                "photo_to_art_ec": _ec(_safe_float(photo_to_art.get("clip_style")), _safe_float(photo_to_art.get("content_lpips"))),
                "summary_path": str(summary_path),
            }
        )
    rows.sort(key=lambda x: x["num_steps"])
    return rows


def summarize_lambda_grid(output_root: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    status_rows = _read_csv(output_root / "lambda_grid" / "status.csv")
    final_rows: list[dict[str, Any]] = []
    best_rows: list[dict[str, Any]] = []
    for status_row in status_rows:
        exp_name = status_row["experiment"]
        batch_summary_path = output_root / "lambda_grid" / "eval" / exp_name / "batch_summary.csv"
        batch_rows = _read_csv(batch_summary_path)
        parsed_rows: list[dict[str, Any]] = []
        for row in batch_rows:
            transfer_style = _safe_float(row.get("transfer_clip_style") or row.get("clip_style_transfer"))
            transfer_lpips = _safe_float(row.get("transfer_content_lpips") or row.get("content_lpips_transfer"))
            transfer_content = _safe_float(row.get("transfer_clip_content") or row.get("clip_content_transfer"))
            all_style = _safe_float(row.get("all_clip_style") or row.get("clip_style_all") or row.get("clip_style"))
            all_lpips = _safe_float(row.get("all_content_lpips") or row.get("content_lpips_all") or row.get("content_lpips"))
            all_content = _safe_float(row.get("all_clip_content") or row.get("clip_content_all") or row.get("clip_content"))
            parsed_rows.append(
                {
                    "experiment": exp_name,
                    "epoch": int(str(row["epoch"]).replace("epoch_", "")),
                    "w_kinetic": _safe_float(status_row["w_kinetic"]),
                    "terminal_swd_weight": _safe_float(status_row["terminal_swd_weight"]),
                    "all_clip_style": all_style,
                    "all_clip_content": all_content,
                    "all_content_lpips": all_lpips,
                    "all_ec": _ec(all_style, all_lpips),
                    "transfer_clip_style": transfer_style,
                    "transfer_clip_content": transfer_content,
                    "transfer_content_lpips": transfer_lpips,
                    "transfer_ec": _ec(transfer_style, transfer_lpips),
                    "photo_to_art_clip_style": _safe_float(row.get("clip_style_photo_to_art")),
                    "photo_to_art_clip_content": _safe_float(row.get("clip_content_photo_to_art")),
                    "photo_to_art_content_lpips": _safe_float(row.get("content_lpips_photo_to_art")),
                    "photo_to_art_ec": _ec(_safe_float(row.get("clip_style_photo_to_art")), _safe_float(row.get("content_lpips_photo_to_art"))),
                    "checkpoint_path": row.get("checkpoint_path"),
                    "output_dir": row.get("output_dir"),
                }
            )
        parsed_rows.sort(key=lambda x: x["epoch"])
        final_rows.append(parsed_rows[-1])
        best_rows.append(max(parsed_rows, key=lambda x: x["transfer_ec"] if x["transfer_ec"] is not None else float("-inf")))
    final_rows.sort(key=lambda x: (x["w_kinetic"], x["terminal_swd_weight"]))
    best_rows.sort(key=lambda x: (x["w_kinetic"], x["terminal_swd_weight"]))
    return final_rows, best_rows


def summarize_efficiency(output_root: Path, write_root: Path) -> list[dict[str, Any]]:
    eff_path = _ensure_efficiency_profile(output_root, write_root)
    payload = _load_json(eff_path)
    rows: list[dict[str, Any]] = []
    for rec in payload.get("records", []):
        row = {
            "batch_size": rec.get("batch_size"),
            "num_steps": rec.get("num_steps"),
            "avg_sec_per_iter": rec.get("avg_sec_per_iter"),
            "avg_sec_per_img": rec.get("avg_sec_per_img"),
            "throughput_img_per_sec": rec.get("throughput_img_per_sec"),
            "peak_vram_mb": rec.get("peak_vram_mb"),
            "peak_reserved_mb": rec.get("peak_reserved_mb"),
            "params": payload.get("params"),
            "macs": payload.get("macs"),
            "flops": payload.get("flops"),
            "flops_backend": payload.get("flops_backend"),
            "measurement_scope": payload.get("measurement_scope"),
        }
        rows.append(row)
    rows.sort(key=lambda x: (x["batch_size"], x["num_steps"]))
    return rows


def main() -> int:
    output_root = _pick_output_root()
    aggregate_root = DEFAULT_WRITE_ROOT
    aggregate_root.mkdir(parents=True, exist_ok=True)
    step_rows = summarize_step_sweep(output_root)
    lambda_final_rows, lambda_best_rows = summarize_lambda_grid(output_root)
    eff_rows = summarize_efficiency(output_root, aggregate_root)

    _write_csv(
        aggregate_root / "step_sweep_pareto.csv",
        step_rows,
        [
            "num_steps",
            "eval_elapsed_sec",
            "end_to_end_img_per_sec",
            "all_clip_style",
            "all_clip_content",
            "all_content_lpips",
            "all_ec",
            "transfer_clip_style",
            "transfer_clip_content",
            "transfer_content_lpips",
            "transfer_ec",
            "photo_to_art_clip_style",
            "photo_to_art_clip_content",
            "photo_to_art_content_lpips",
            "photo_to_art_ec",
            "summary_path",
        ],
    )
    _write_csv(
        aggregate_root / "lambda_grid_final_epoch.csv",
        lambda_final_rows,
        [
            "experiment",
            "epoch",
            "w_kinetic",
            "terminal_swd_weight",
            "all_clip_style",
            "all_clip_content",
            "all_content_lpips",
            "all_ec",
            "transfer_clip_style",
            "transfer_clip_content",
            "transfer_content_lpips",
            "transfer_ec",
            "photo_to_art_clip_style",
            "photo_to_art_clip_content",
            "photo_to_art_content_lpips",
            "photo_to_art_ec",
            "checkpoint_path",
            "output_dir",
        ],
    )
    _write_csv(
        aggregate_root / "lambda_grid_best_transfer_ec.csv",
        lambda_best_rows,
        [
            "experiment",
            "epoch",
            "w_kinetic",
            "terminal_swd_weight",
            "all_clip_style",
            "all_clip_content",
            "all_content_lpips",
            "all_ec",
            "transfer_clip_style",
            "transfer_clip_content",
            "transfer_content_lpips",
            "transfer_ec",
            "photo_to_art_clip_style",
            "photo_to_art_clip_content",
            "photo_to_art_content_lpips",
            "photo_to_art_ec",
            "checkpoint_path",
            "output_dir",
        ],
    )
    _write_csv(
        aggregate_root / "efficiency_profile.csv",
        eff_rows,
        [
            "batch_size",
            "num_steps",
            "avg_sec_per_iter",
            "avg_sec_per_img",
            "throughput_img_per_sec",
            "peak_vram_mb",
            "peak_reserved_mb",
            "params",
            "macs",
            "flops",
            "flops_backend",
            "measurement_scope",
        ],
    )
    _write_json(
        aggregate_root / "summary_manifest.json",
        {
            "output_root": str(output_root),
            "step_rows": len(step_rows),
            "lambda_final_rows": len(lambda_final_rows),
            "lambda_best_rows": len(lambda_best_rows),
            "efficiency_rows": len(eff_rows),
            "files": {
                "step_sweep_pareto": str(aggregate_root / "step_sweep_pareto.csv"),
                "lambda_grid_final_epoch": str(aggregate_root / "lambda_grid_final_epoch.csv"),
                "lambda_grid_best_transfer_ec": str(aggregate_root / "lambda_grid_best_transfer_ec.csv"),
                "efficiency_profile": str(aggregate_root / "efficiency_profile.csv"),
            },
        },
    )
    print(f"Aggregates written to: {aggregate_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
