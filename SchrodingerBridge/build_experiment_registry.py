from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _train_record(path: Path) -> dict[str, Any] | None:
    config = _read_json(path / "config.json")
    if config is None:
        return None
    ablation = config.get("ablation", {})
    training = config.get("training", {})
    bridge = config.get("bridge", {})
    checkpoints = sorted(path.glob("epoch_*.pt"))
    return {
        "kind": "train",
        "dir": path.name,
        "path": str(path),
        "linked_train_dir": path.name,
        "linked_eval_dir": "",
        "suite": "",
        "ablation_name": ablation.get("name", ""),
        "axis": ablation.get("axis", ""),
        "notes": ablation.get("notes", ""),
        "epochs": training.get("num_epochs", ""),
        "batch_size": training.get("batch_size", ""),
        "learning_rate": training.get("learning_rate", ""),
        "terminal_swd_weight": bridge.get("terminal_swd_weight", ""),
        "w_kinetic": bridge.get("w_kinetic", ""),
        "w_cycle": bridge.get("w_cycle", ""),
        "w_repulsive": bridge.get("w_repulsive", ""),
        "w_low_freq": bridge.get("w_low_freq", ""),
        "swd_use_high_freq": bridge.get("swd_use_high_freq", ""),
        "checkpoint_count": len(checkpoints),
        "latest_checkpoint": checkpoints[-1].name if checkpoints else "",
        "checkpoint": "",
        "style_transfer_clip_style": "",
        "style_transfer_clip_content": "",
        "style_transfer_lpips": "",
        "photo_to_art_clip_style": "",
        "photo_to_art_clip_content": "",
        "photo_to_art_lpips": "",
    }


def _eval_record(path: Path) -> dict[str, Any] | None:
    summary = _read_json(path / "summary.json")
    if summary is None:
        return None
    checkpoint_ref = str(summary.get("checkpoint", ""))
    train_dir = Path(checkpoint_ref).parts[0] if checkpoint_ref else ""
    analysis = summary.get("analysis", {})
    sta = analysis.get("style_transfer_ability", summary.get("style_transfer_ability", {}))
    p2a = analysis.get("photo_to_art_performance", summary.get("photo_to_art_performance", {}))
    return {
        "kind": "eval",
        "dir": path.name,
        "path": str(path),
        "linked_train_dir": train_dir,
        "linked_eval_dir": path.name,
        "suite": "",
        "ablation_name": "",
        "axis": "",
        "notes": "",
        "epochs": "",
        "batch_size": "",
        "learning_rate": "",
        "terminal_swd_weight": "",
        "w_kinetic": "",
        "w_cycle": "",
        "w_repulsive": "",
        "w_low_freq": "",
        "swd_use_high_freq": "",
        "checkpoint_count": "",
        "latest_checkpoint": "",
        "checkpoint": checkpoint_ref,
        "style_transfer_clip_style": sta.get("clip_style", ""),
        "style_transfer_clip_content": sta.get("clip_content", ""),
        "style_transfer_lpips": sta.get("content_lpips", ""),
        "photo_to_art_clip_style": p2a.get("clip_style", ""),
        "photo_to_art_clip_content": p2a.get("clip_content", ""),
        "photo_to_art_lpips": p2a.get("content_lpips", ""),
    }


def _debug_report_record(path: Path) -> dict[str, Any] | None:
    report = _read_json(path)
    if report is None:
        return None
    metrics = report.get("metrics", {})
    decoded = report.get("decoded_artifact_stats", {}) or {}
    attn = report.get("semantic_attention_diagnostics", {}) or {}
    return {
        "kind": "debug_report",
        "dir": path.stem,
        "path": str(path),
        "linked_train_dir": Path(str(report.get("resume_checkpoint", ""))).parts[0] if report.get("resume_checkpoint") else "",
        "linked_eval_dir": "",
        "suite": "orthogonal_phase_space_sweep_debug",
        "ablation_name": "",
        "axis": "debug",
        "notes": "",
        "epochs": "",
        "batch_size": "",
        "learning_rate": "",
        "terminal_swd_weight": "",
        "w_kinetic": "",
        "w_cycle": "",
        "w_repulsive": "",
        "w_low_freq": "",
        "swd_use_high_freq": "",
        "checkpoint_count": "",
        "latest_checkpoint": "",
        "checkpoint": str(report.get("resume_checkpoint", "")),
        "style_transfer_clip_style": "",
        "style_transfer_clip_content": "",
        "style_transfer_lpips": "",
        "photo_to_art_clip_style": "",
        "photo_to_art_clip_content": "",
        "photo_to_art_lpips": "",
        "debug_loss": metrics.get("loss", ""),
        "debug_terminal_swd": metrics.get("terminal_swd", ""),
        "debug_kinetic": metrics.get("kinetic_energy", ""),
        "debug_dec_out_grad_swd": ((report.get("component_grad_balance", {}) or {}).get("terminal_swd", {}) or {}).get("dec_out.weight", ""),
        "debug_attn_top1": attn.get("mean_top1_prob", ""),
        "debug_attn_entropy_norm": attn.get("normalized_entropy", ""),
        "debug_darkest_sample": decoded.get("darkest_sample_index", ""),
        "debug_darkest_ratio": decoded.get("darkest_sample_ratio", ""),
    }


def _scan_one_dir(base: Path, rows: list[dict[str, Any]]) -> None:
    if not base.exists():
        return
    for child in sorted(base.iterdir()):
        if not child.is_dir():
            continue
        if child.name.startswith("o") and (child / "config.json").exists():
            rec = _train_record(child)
            if rec:
                rows.append(rec)
        elif child.name.startswith("o") and (child / "summary.json").exists():
            rec = _eval_record(child)
            if rec:
                rows.append(rec)


def build_registry(repo: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    _scan_one_dir(repo, rows)
    _scan_one_dir(repo / "exp" / "runs", rows)

    debug_dir = repo / "orthogonal_phase_space_sweep_debug" / "reports"
    if not debug_dir.exists():
        debug_dir = repo / "exp" / "configs" / "orthogonal_phase_space_sweep_debug" / "reports"
    if debug_dir.exists():
        for report_path in sorted(debug_dir.glob("*.json")):
            rec = _debug_report_record(report_path)
            if rec:
                rows.append(rec)
    return rows


def main() -> None:
    repo = Path(__file__).resolve().parent
    out_dir = repo / "docs" / "experiments"
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = build_registry(repo)
    json_path = out_dir / "experiment_registry.json"
    csv_path = out_dir / "experiment_registry.csv"

    json_path.write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
    fieldnames: list[str] = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(json_path)
    print(csv_path)


if __name__ == "__main__":
    main()
