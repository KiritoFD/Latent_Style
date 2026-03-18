#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any


MODEL_KEYS = [
    "use_decoder_spatial_inject",
    "use_delta_highpass_bias",
    "use_style_delta_gate",
    "use_decoder_adagn",
    "use_style_spatial_highpass",
    "normalize_style_spatial_maps",
    "use_output_style_affine",
    "use_style_spatial_blur",
    "style_ref_gain",
    "style_spatial_pre_gain_16",
    "style_spatial_dec_gain_32",
    "style_texture_gain",
    "style_delta_lowfreq_gain",
    "highpass_last_step_scale",
    "style_gate_floor",
    "style_texture_mode",
    "style_strength_step_curve",
]

LOSS_KEYS = [
    "w_distill",
    "w_code",
    "w_struct",
    "w_edge",
    "w_cycle",
    "w_stroke_gram",
    "w_color_moment",
    "w_style_spatial_tv",
    "w_nce",
    "w_push",
    "w_delta_tv",
    "w_semigroup",
    "distill_low_only",
    "distill_cross_domain_only",
    "train_num_steps_min",
    "train_num_steps_max",
    "train_style_strength_min",
    "train_style_strength_max",
    "teacher_interval_steps",
    "stroke_interval_steps",
    "semigroup_interval_steps",
]


def safe_get(d: dict[str, Any], path: list[str], default: Any = None) -> Any:
    cur: Any = d
    for key in path:
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def file_sha1(path: Path) -> str | None:
    if not path.exists() or not path.is_file():
        return None
    h = hashlib.sha1()
    with path.open("rb") as f:
        while True:
            chunk = f.read(1024 * 1024)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def parse_epoch_dir_name(name: str) -> int | None:
    if not name.startswith("epoch_"):
        if name.isdigit():
            return int(name)
        return None
    part = name[len("epoch_") :]
    if part.isdigit():
        return int(part)
    return None


def run_family(run_name: str, rel_path: str) -> str:
    n = run_name.lower()
    rel = rel_path.lower()
    if rel.startswith("experiments\\small-exp") or rel.startswith("experiments/small-exp"):
        return "small-exp"
    if n.startswith("overfit50") or "\\overfit50" in rel or "/overfit50" in rel:
        return "overfit50"
    if n.startswith("full_300"):
        return "full_300"
    if n.startswith("full_250"):
        return "full_250"
    if n.startswith("full-300"):
        return "full-300"
    if n.startswith("full_"):
        return "full_other"
    if n.startswith("adacut"):
        return "adacut"
    if rel.startswith("experiments\\") or rel.startswith("experiments/"):
        return "experiments_misc"
    return "other"


def is_run_dir(path: Path) -> bool:
    if not path.is_dir():
        return False
    if (path / "full_eval").is_dir():
        return True
    if (path / "logs").is_dir() and any((path / "logs").glob("training_*.csv")):
        return True
    if any(path.glob("epoch_*.pt")):
        return True
    return False


def discover_runs(root: Path) -> list[Path]:
    runs: list[Path] = []
    for dirpath, dirnames, _filenames in os.walk(root):
        current = Path(dirpath)
        if current.name.startswith("src_snapshot_"):
            dirnames[:] = []
            continue
        if current.name == "torch_compile_cache":
            dirnames[:] = []
            continue
        if is_run_dir(current):
            runs.append(current)
            dirnames[:] = [d for d in dirnames if not d.startswith("src_snapshot_")]
    runs = sorted(set(runs))
    return runs


def parse_training_csv(csv_path: Path) -> dict[str, Any]:
    last_row: dict[str, str] | None = None
    header: list[str] = []
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames:
            header = list(reader.fieldnames)
        for row in reader:
            last_row = row
    out: dict[str, Any] = {
        "csv_path": str(csv_path),
        "rows": 0 if last_row is None else 1,
        "header": header,
        "last_epoch": None,
        "last_loss": None,
        "last_lr": None,
    }
    if last_row is None:
        return out
    try:
        out["last_epoch"] = int(float(last_row.get("epoch", "")))
    except Exception:
        out["last_epoch"] = None
    for key, target in [("loss", "last_loss"), ("lr", "last_lr")]:
        v = last_row.get(key)
        if v is None or v == "":
            out[target] = None
            continue
        try:
            out[target] = float(v)
        except Exception:
            out[target] = None
    return out


def load_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def float_or_none(x: Any) -> float | None:
    if x is None:
        return None
    try:
        v = float(x)
    except Exception:
        return None
    if math.isnan(v) or math.isinf(v):
        return None
    return v


@dataclass
class RunRecord:
    record: dict[str, Any]
    history_rows: list[dict[str, Any]]
    snapshot_rows: list[dict[str, Any]]


def analyze_run(root: Path, run_dir: Path) -> RunRecord:
    rel_path = run_dir.relative_to(root)
    run_name = run_dir.name
    family = run_family(run_name, str(rel_path))

    full_eval_dir = run_dir / "full_eval"
    full_eval_exists = full_eval_dir.is_dir()

    latest_epoch: int | None = None
    latest_summary_path: Path | None = None
    latest_summary: dict[str, Any] | None = None

    if full_eval_exists:
        for p in full_eval_dir.glob("*/summary.json"):
            ep = parse_epoch_dir_name(p.parent.name)
            if ep is None:
                continue
            if latest_epoch is None or ep > latest_epoch:
                latest_epoch = ep
                latest_summary_path = p
        if latest_summary_path is not None:
            latest_summary = load_json(latest_summary_path)

    summary_history_path = full_eval_dir / "summary_history.json"
    summary_history = load_json(summary_history_path)

    history_rows: list[dict[str, Any]] = []
    if isinstance(summary_history, dict):
        for row in summary_history.get("rounds", []) or []:
            if not isinstance(row, dict):
                continue
            history_rows.append(
                {
                    "run": run_name,
                    "rel_path": str(rel_path).replace("\\", "/"),
                    "family": family,
                    "epoch": row.get("epoch"),
                    "transfer_clip_style": float_or_none(row.get("transfer_clip_style")),
                    "transfer_content_lpips": float_or_none(row.get("transfer_content_lpips")),
                    "transfer_classifier_acc": float_or_none(row.get("transfer_classifier_acc")),
                    "photo_to_art_clip_style": float_or_none(row.get("photo_to_art_clip_style")),
                    "photo_to_art_classifier_acc": float_or_none(row.get("photo_to_art_classifier_acc")),
                }
            )

    logs_dir = run_dir / "logs"
    train_csvs = sorted(logs_dir.glob("training_*.csv")) if logs_dir.is_dir() else []
    latest_train = parse_training_csv(train_csvs[-1]) if train_csvs else None

    snapshots = sorted([p for p in run_dir.glob("src_snapshot_*") if p.is_dir()])
    latest_snapshot = snapshots[-1] if snapshots else None

    snapshot_rows: list[dict[str, Any]] = []
    snapshot_model_hashes: set[str] = set()
    snapshot_loss_hashes: set[str] = set()
    snapshot_trainer_hashes: set[str] = set()
    for snap in snapshots:
        snap_cfg = None
        snap_cfg_path: Path | None = None
        for cand in ["config.json", "config_overfit.json", "overfit50.json"]:
            cpath = snap / cand
            if cpath.exists():
                snap_cfg = load_json(cpath)
                if snap_cfg is not None:
                    snap_cfg_path = cpath
                    break
        snap_model_cfg = snap_cfg.get("model", {}) if isinstance(snap_cfg, dict) else {}
        snap_loss_cfg = snap_cfg.get("loss", {}) if isinstance(snap_cfg, dict) else {}
        snap_model_hash = file_sha1(snap / "model.py")
        snap_loss_hash = file_sha1(snap / "losses.py")
        snap_trainer_hash = file_sha1(snap / "trainer.py")
        if snap_model_hash:
            snapshot_model_hashes.add(snap_model_hash)
        if snap_loss_hash:
            snapshot_loss_hashes.add(snap_loss_hash)
        if snap_trainer_hash:
            snapshot_trainer_hashes.add(snap_trainer_hash)
        snap_row: dict[str, Any] = {
            "run": run_name,
            "rel_path": str(rel_path).replace("\\", "/"),
            "family": family,
            "snapshot_name": snap.name,
            "snapshot_path": str(snap),
            "snapshot_config_path": None if snap_cfg_path is None else str(snap_cfg_path),
            "snapshot_model_sha1": snap_model_hash,
            "snapshot_losses_sha1": snap_loss_hash,
            "snapshot_trainer_sha1": snap_trainer_hash,
        }
        for k in MODEL_KEYS:
            snap_row[f"model__{k}"] = snap_model_cfg.get(k)
        for k in LOSS_KEYS:
            snap_row[f"loss__{k}"] = snap_loss_cfg.get(k)
        snapshot_rows.append(snap_row)

    snapshot_cfg = None
    if latest_snapshot is not None:
        for cand in ["config.json", "config_overfit.json", "overfit50.json"]:
            cpath = latest_snapshot / cand
            if cpath.exists():
                snapshot_cfg = load_json(cpath)
                if snapshot_cfg is not None:
                    break

    model_cfg = snapshot_cfg.get("model", {}) if isinstance(snapshot_cfg, dict) else {}
    loss_cfg = snapshot_cfg.get("loss", {}) if isinstance(snapshot_cfg, dict) else {}

    cfg_extract: dict[str, Any] = {}
    for k in MODEL_KEYS:
        cfg_extract[f"model__{k}"] = model_cfg.get(k)
    for k in LOSS_KEYS:
        cfg_extract[f"loss__{k}"] = loss_cfg.get(k)

    best_transfer = None
    best_transfer_epoch = None
    best_p2a = None
    best_cls = None
    if isinstance(summary_history, dict):
        best_transfer_block = safe_get(summary_history, ["best", "best_transfer_clip_style"], None)
        if isinstance(best_transfer_block, dict):
            best_transfer = float_or_none(best_transfer_block.get("transfer_clip_style"))
            best_transfer_epoch = best_transfer_block.get("epoch")
        best_p2a_block = safe_get(summary_history, ["best", "best_photo_to_art_clip_style"], None)
        if isinstance(best_p2a_block, dict):
            best_p2a = float_or_none(best_p2a_block.get("photo_to_art_clip_style"))
        best_cls_block = safe_get(summary_history, ["best", "best_transfer_classifier_acc"], None)
        if isinstance(best_cls_block, dict):
            best_cls = float_or_none(best_cls_block.get("transfer_classifier_acc"))

    matrix_breakdown = latest_summary.get("matrix_breakdown", {}) if isinstance(latest_summary, dict) else {}
    cross_pairs: list[dict[str, Any]] = []
    if isinstance(matrix_breakdown, dict):
        for src_name, src_block in matrix_breakdown.items():
            if not isinstance(src_block, dict):
                continue
            for tgt_name, tgt_block in src_block.items():
                if not isinstance(tgt_block, dict):
                    continue
                cross_pairs.append(
                    {
                        "src": src_name,
                        "tgt": tgt_name,
                        "count": float_or_none(tgt_block.get("count")),
                        "clip_style": float_or_none(tgt_block.get("clip_style")),
                        "clip_content": float_or_none(tgt_block.get("clip_content")),
                        "content_lpips": float_or_none(tgt_block.get("content_lpips")),
                        "classifier_acc": float_or_none(tgt_block.get("classifier_acc")),
                    }
                )

    offdiag = [p for p in cross_pairs if p["src"] != p["tgt"] and p["clip_style"] is not None]
    offdiag_mean = None
    if offdiag:
        offdiag_mean = sum(float(p["clip_style"]) for p in offdiag) / len(offdiag)
    style_names = set()
    for p in cross_pairs:
        style_names.add(str(p["src"]))
        style_names.add(str(p["tgt"]))
    matrix_style_count = len(style_names)
    matrix_pair_count = len(cross_pairs)
    matrix_offdiag_count = len([p for p in cross_pairs if p["src"] != p["tgt"]])
    matrix_complete_square = bool(
        matrix_style_count >= 2 and matrix_pair_count >= (matrix_style_count * matrix_style_count)
    )
    pair_counts = [float(p["count"]) for p in cross_pairs if p.get("count") is not None]
    matrix_eval_count_mean = (sum(pair_counts) / len(pair_counts)) if pair_counts else None
    matrix_eval_count_min = min(pair_counts) if pair_counts else None
    matrix_eval_count_max = max(pair_counts) if pair_counts else None

    latest_transfer = None
    latest_p2a = None
    latest_cls = None
    latest_content_lpips = None
    latest_overall_acc = None
    if isinstance(latest_summary, dict):
        latest_transfer = float_or_none(safe_get(latest_summary, ["analysis", "style_transfer_ability", "clip_style"]))
        latest_p2a = float_or_none(safe_get(latest_summary, ["analysis", "photo_to_art_performance", "clip_style"]))
        latest_cls = float_or_none(safe_get(latest_summary, ["analysis", "style_transfer_ability", "classifier_acc"]))
        latest_content_lpips = float_or_none(
            safe_get(latest_summary, ["analysis", "style_transfer_ability", "content_lpips"])
        )
        latest_overall_acc = float_or_none(safe_get(latest_summary, ["classification_report", "accuracy"]))

    if best_transfer is None:
        best_transfer = latest_transfer
        best_transfer_epoch = latest_epoch
    if best_p2a is None:
        best_p2a = latest_p2a
    if best_cls is None:
        best_cls = latest_cls

    model_hash = file_sha1(latest_snapshot / "model.py") if latest_snapshot is not None else None
    loss_hash = file_sha1(latest_snapshot / "losses.py") if latest_snapshot is not None else None
    trainer_hash = file_sha1(latest_snapshot / "trainer.py") if latest_snapshot is not None else None

    record: dict[str, Any] = {
        "run": run_name,
        "rel_path": str(rel_path).replace("\\", "/"),
        "family": family,
        "has_full_eval": full_eval_exists,
        "latest_epoch": latest_epoch,
        "latest_summary_path": None if latest_summary_path is None else str(latest_summary_path),
        "history_path": str(summary_history_path) if summary_history_path.exists() else None,
        "history_rounds": len(history_rows),
        "latest_transfer_clip_style": latest_transfer,
        "latest_photo_to_art_clip_style": latest_p2a,
        "latest_transfer_classifier_acc": latest_cls,
        "latest_transfer_content_lpips": latest_content_lpips,
        "latest_classification_accuracy": latest_overall_acc,
        "latest_offdiag_clip_style_mean": offdiag_mean,
        "matrix_style_count": matrix_style_count,
        "matrix_pair_count": matrix_pair_count,
        "matrix_offdiag_count": matrix_offdiag_count,
        "matrix_complete_square": matrix_complete_square,
        "matrix_eval_count_mean": matrix_eval_count_mean,
        "matrix_eval_count_min": matrix_eval_count_min,
        "matrix_eval_count_max": matrix_eval_count_max,
        "best_transfer_clip_style": best_transfer,
        "best_transfer_clip_style_epoch": best_transfer_epoch,
        "best_photo_to_art_clip_style": best_p2a,
        "best_transfer_classifier_acc": best_cls,
        "train_csv_count": len(train_csvs),
        "latest_train_csv": None if latest_train is None else latest_train["csv_path"],
        "latest_train_epoch": None if latest_train is None else latest_train["last_epoch"],
        "latest_train_loss": None if latest_train is None else latest_train["last_loss"],
        "latest_train_lr": None if latest_train is None else latest_train["last_lr"],
        "checkpoint_count": len(list(run_dir.glob("epoch_*.pt"))),
        "snapshot_count": len(snapshots),
        "latest_snapshot": None if latest_snapshot is None else str(latest_snapshot),
        "latest_snapshot_model_sha1": model_hash,
        "latest_snapshot_losses_sha1": loss_hash,
        "latest_snapshot_trainer_sha1": trainer_hash,
        "snapshot_model_hash_count": len(snapshot_model_hashes),
        "snapshot_losses_hash_count": len(snapshot_loss_hashes),
        "snapshot_trainer_hash_count": len(snapshot_trainer_hashes),
        "config_excerpt": cfg_extract,
    }
    return RunRecord(record=record, history_rows=history_rows, snapshot_rows=snapshot_rows)


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for row in rows:
            w.writerow(row)


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze experiments-cycle runs.")
    parser.add_argument("--root", type=Path, default=Path("experiments-cycle"))
    parser.add_argument("--out-dir", type=Path, default=Path("docs/experiments_cycle/data"))
    args = parser.parse_args()

    root = args.root.resolve()
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    runs = discover_runs(root)

    run_records: list[dict[str, Any]] = []
    history_rows: list[dict[str, Any]] = []
    snapshot_rows: list[dict[str, Any]] = []
    for run_dir in runs:
        analyzed = analyze_run(root, run_dir)
        run_records.append(analyzed.record)
        history_rows.extend(analyzed.history_rows)
        snapshot_rows.extend(analyzed.snapshot_rows)

    def sort_metric_key(row: dict[str, Any]) -> tuple[float, float]:
        best = row.get("best_transfer_clip_style")
        latest = row.get("latest_transfer_clip_style")
        b = -1.0 if best is None else float(best)
        l = -1.0 if latest is None else float(latest)
        return (b, l)

    run_records.sort(key=sort_metric_key, reverse=True)

    json_path = out_dir / "runs_detailed.json"
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(run_records, f, indent=2, ensure_ascii=False)

    flat_rows: list[dict[str, Any]] = []
    for rec in run_records:
        flat = {k: v for k, v in rec.items() if k != "config_excerpt"}
        flat.update(rec.get("config_excerpt", {}))
        flat_rows.append(flat)

    base_fields = [
        "run",
        "rel_path",
        "family",
        "has_full_eval",
        "latest_epoch",
        "latest_transfer_clip_style",
        "best_transfer_clip_style",
        "best_transfer_clip_style_epoch",
        "latest_photo_to_art_clip_style",
        "best_photo_to_art_clip_style",
        "latest_transfer_classifier_acc",
        "best_transfer_classifier_acc",
        "latest_transfer_content_lpips",
        "latest_classification_accuracy",
        "latest_offdiag_clip_style_mean",
        "matrix_style_count",
        "matrix_pair_count",
        "matrix_offdiag_count",
        "matrix_complete_square",
        "matrix_eval_count_mean",
        "matrix_eval_count_min",
        "matrix_eval_count_max",
        "history_rounds",
        "train_csv_count",
        "latest_train_epoch",
        "latest_train_loss",
        "latest_train_lr",
        "checkpoint_count",
        "snapshot_count",
        "latest_snapshot",
        "latest_snapshot_model_sha1",
        "latest_snapshot_losses_sha1",
        "latest_snapshot_trainer_sha1",
        "snapshot_model_hash_count",
        "snapshot_losses_hash_count",
        "snapshot_trainer_hash_count",
    ]
    cfg_fields = [f"model__{k}" for k in MODEL_KEYS] + [f"loss__{k}" for k in LOSS_KEYS]
    csv_fields = base_fields + cfg_fields
    write_csv(out_dir / "runs_metrics.csv", flat_rows, csv_fields)

    hist_fields = [
        "run",
        "rel_path",
        "family",
        "epoch",
        "transfer_clip_style",
        "transfer_content_lpips",
        "transfer_classifier_acc",
        "photo_to_art_clip_style",
        "photo_to_art_classifier_acc",
    ]
    write_csv(out_dir / "history_rounds.csv", history_rows, hist_fields)

    snap_fields = [
        "run",
        "rel_path",
        "family",
        "snapshot_name",
        "snapshot_path",
        "snapshot_config_path",
        "snapshot_model_sha1",
        "snapshot_losses_sha1",
        "snapshot_trainer_sha1",
    ] + [f"model__{k}" for k in MODEL_KEYS] + [f"loss__{k}" for k in LOSS_KEYS]
    write_csv(out_dir / "snapshot_timeline.csv", snapshot_rows, snap_fields)

    family_map: dict[str, list[dict[str, Any]]] = {}
    for r in run_records:
        family_map.setdefault(str(r.get("family", "other")), []).append(r)

    family_rows: list[dict[str, Any]] = []
    for fam, items in sorted(family_map.items()):
        with_eval = [x for x in items if x.get("has_full_eval")]
        with_metric = [x for x in items if x.get("best_transfer_clip_style") is not None]
        top = sorted(with_metric, key=lambda x: float(x["best_transfer_clip_style"]), reverse=True)
        family_rows.append(
            {
                "family": fam,
                "run_count": len(items),
                "with_full_eval": len(with_eval),
                "with_best_metric": len(with_metric),
                "mean_best_transfer_clip_style": (
                    None
                    if not with_metric
                    else sum(float(x["best_transfer_clip_style"]) for x in with_metric) / len(with_metric)
                ),
                "max_best_transfer_clip_style": None if not top else top[0]["best_transfer_clip_style"],
                "max_best_transfer_run": None if not top else top[0]["run"],
            }
        )
    write_csv(
        out_dir / "family_summary.csv",
        family_rows,
        [
            "family",
            "run_count",
            "with_full_eval",
            "with_best_metric",
            "mean_best_transfer_clip_style",
            "max_best_transfer_clip_style",
            "max_best_transfer_run",
        ],
    )

    print(f"Runs discovered: {len(runs)}")
    print(f"Wrote: {json_path}")
    print(f"Wrote: {out_dir / 'runs_metrics.csv'}")
    print(f"Wrote: {out_dir / 'history_rounds.csv'}")
    print(f"Wrote: {out_dir / 'snapshot_timeline.csv'}")
    print(f"Wrote: {out_dir / 'family_summary.csv'}")


if __name__ == "__main__":
    main()
