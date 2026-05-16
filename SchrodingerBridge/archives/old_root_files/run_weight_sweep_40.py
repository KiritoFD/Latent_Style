from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
SWEEP_ROOT = ROOT / "weight_sweep_40"
MANIFEST = SWEEP_ROOT / "manifest.csv"
SAMST_SUMMARY = ROOT.parent / "Related_Works" / "run_511" / "complete_750" / "samst_strict" / "summary.json"


def _run(cmd: list[str], *, cwd: Path = ROOT, log_path: Path | None = None) -> int:
    print(" ".join(str(x) for x in cmd), flush=True)
    if log_path is None:
        return subprocess.run(cmd, cwd=cwd).returncode
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8", errors="replace") as f:
        f.write("\n\n$ " + " ".join(str(x) for x in cmd) + "\n")
        f.flush()
        return subprocess.run(cmd, cwd=cwd, stdout=f, stderr=subprocess.STDOUT).returncode


def _read_manifest() -> list[dict[str, str]]:
    if not MANIFEST.exists():
        rc = _run([sys.executable, str(ROOT / "prepare_weight_sweep_40.py")], cwd=ROOT)
        if rc != 0:
            raise SystemExit(rc)
    with MANIFEST.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _safe_float(value: Any) -> float | None:
    try:
        if value is None or value == "":
            return None
        return float(value)
    except Exception:
        return None


def _score(style: float | None, lpips: float | None, *, style_weight: float = 0.65) -> float | None:
    if style is None or lpips is None:
        return None
    inv_lpips = 1.0 - lpips
    if style_weight < 0:
        return style * inv_lpips
    return style_weight * style + (1.0 - style_weight) * inv_lpips


def _summary_metrics(summary_path: Path) -> dict[str, Any]:
    payload = _load_json(summary_path)
    analysis = payload.get("analysis", {}) or {}
    all_pairs = analysis.get("all_pairs_overview", {}) or {}
    transfer = analysis.get("style_transfer_ability", {}) or {}
    photo = analysis.get("photo_to_art_performance", {}) or {}
    identity = analysis.get("identity_reconstruction", {}) or {}
    out: dict[str, Any] = {
        "summary_path": str(summary_path),
        "checkpoint": payload.get("checkpoint", ""),
        "all_clip_style": all_pairs.get("clip_style"),
        "all_clip_content": all_pairs.get("clip_content"),
        "all_content_lpips": all_pairs.get("content_lpips"),
        "transfer_clip_style": transfer.get("clip_style"),
        "transfer_clip_content": transfer.get("clip_content"),
        "transfer_content_lpips": transfer.get("content_lpips"),
        "photo_clip_style": photo.get("clip_style"),
        "photo_clip_content": photo.get("clip_content"),
        "photo_content_lpips": photo.get("content_lpips"),
        "identity_clip_style": identity.get("clip_style"),
        "identity_clip_content": identity.get("clip_content"),
        "identity_content_lpips": identity.get("content_lpips"),
    }
    out["score_ec_all"] = _score(out["all_clip_style"], out["all_content_lpips"], style_weight=-1)
    out["score_65_35_all"] = _score(out["all_clip_style"], out["all_content_lpips"], style_weight=0.65)
    out["score_ec_transfer"] = _score(out["transfer_clip_style"], out["transfer_content_lpips"], style_weight=-1)
    out["score_65_35_transfer"] = _score(out["transfer_clip_style"], out["transfer_content_lpips"], style_weight=0.65)
    out["score_ec_photo"] = _score(out["photo_clip_style"], out["photo_content_lpips"], style_weight=-1)
    out["score_65_35_photo"] = _score(out["photo_clip_style"], out["photo_content_lpips"], style_weight=0.65)
    return out


def _matrix_rows(summary_path: Path, base: dict[str, Any]) -> list[dict[str, Any]]:
    payload = _load_json(summary_path)
    matrix = payload.get("matrix_breakdown", {}) or {}
    rows: list[dict[str, Any]] = []
    for src, by_target in matrix.items():
        if not isinstance(by_target, dict):
            continue
        for target, cell in by_target.items():
            if not isinstance(cell, dict):
                continue
            row = dict(base)
            row.update(
                {
                    "src": src,
                    "target": target,
                    "is_identity": src == target,
                    "is_transfer": src != target,
                    "is_photo_to_art": src == "photo" and target != "photo",
                    "count": cell.get("count"),
                    "clip_style": cell.get("clip_style"),
                    "clip_content": cell.get("clip_content"),
                    "content_lpips": cell.get("content_lpips"),
                    "clip_dir": cell.get("clip_dir"),
                }
            )
            row["score_ec"] = _score(row["clip_style"], row["content_lpips"], style_weight=-1)
            row["score_65_35"] = _score(row["clip_style"], row["content_lpips"], style_weight=0.65)
            rows.append(row)
    return rows


def _train_one(row: dict[str, str], *, force_train: bool = False) -> dict[str, Any]:
    exp_id = row["experiment_id"]
    run_dir = Path(row["run_dir"])
    final_ckpt = run_dir / "epoch_0008.pt"
    log_path = SWEEP_ROOT / "logs" / f"{exp_id}.train.log"
    if final_ckpt.exists() and not force_train:
        return {"train_status": "skipped_existing", "train_rc": 0, "train_time_sec": ""}
    start = time.time()
    rc = _run([sys.executable, "run.py", "--config", row["config_path"]], cwd=ROOT, log_path=log_path)
    return {"train_status": "ok" if rc == 0 else "failed", "train_rc": rc, "train_time_sec": round(time.time() - start, 3)}


def _eval_one(row: dict[str, str], *, force_eval: bool = False) -> dict[str, Any]:
    exp_id = row["experiment_id"]
    run_dir = Path(row["run_dir"])
    output_root = run_dir / "full_eval"
    log_path = SWEEP_ROOT / "logs" / f"{exp_id}.eval.log"
    batch_summary = output_root / "batch_summary.csv"
    cmd = [
        sys.executable,
        "run_evaluation.py",
        str(run_dir),
        "--output",
        str(output_root),
        "--batch_size",
        "20",
        "--max_src_samples",
        "30",
        "--max_ref_compare",
        "50",
        "--max_ref_cache",
        "256",
        "--ref_feature_batch_size",
        "64",
        "--num_steps",
        "12",
        "--step_size",
        "1.0",
    ]
    if force_eval:
        cmd.append("--force")
    if batch_summary.exists() and not force_eval:
        return {"eval_status": "skipped_existing", "eval_rc": 0, "eval_time_sec": ""}
    start = time.time()
    rc = _run(cmd, cwd=ROOT, log_path=log_path)
    return {"eval_status": "ok" if rc == 0 else "failed", "eval_rc": rc, "eval_time_sec": round(time.time() - start, 3)}


def _training_time_rows(exp_row: dict[str, str]) -> list[dict[str, Any]]:
    run_dir = Path(exp_row["run_dir"])
    logs = sorted((run_dir / "logs").glob("training_*.csv"))
    if not logs:
        return []
    latest = logs[-1]
    rows: list[dict[str, Any]] = []
    with latest.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            rows.append(
                {
                    "experiment_id": exp_row["experiment_id"],
                    "epoch": f"epoch_{int(row.get('epoch', 0)):04d}",
                    "epoch_time_sec": row.get("epoch_time_sec", ""),
                    "samples_per_sec": row.get("samples_per_sec", ""),
                    "loss": row.get("loss", ""),
                    "terminal_swd": row.get("terminal_swd", ""),
                    "kinetic_energy": row.get("kinetic_energy", ""),
                }
            )
    return rows


def _collect(manifest_rows: list[dict[str, str]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    summary_rows: list[dict[str, Any]] = []
    matrix_rows: list[dict[str, Any]] = []
    train_rows: list[dict[str, Any]] = []
    for exp in manifest_rows:
        train_rows.extend(_training_time_rows(exp))
        run_dir = Path(exp["run_dir"])
        for summary_path in sorted((run_dir / "full_eval").glob("epoch_*/summary.json")):
            epoch = summary_path.parent.name
            base = {
                "experiment_id": exp["experiment_id"],
                "epoch": epoch,
                "epoch_num": int(epoch.split("_")[-1]),
                "k_value": exp["k_value"],
                "recipe_id": exp["recipe_id"],
                "description": exp["description"],
                "content_weights": exp["content_weights"],
                "target_weights": exp["target_weights"],
                "balance_target_styles_per_batch": exp["balance_target_styles_per_batch"],
            }
            row = dict(base)
            row.update(_summary_metrics(summary_path))
            summary_rows.append(row)
            matrix_rows.extend(_matrix_rows(summary_path, base))
    return summary_rows, matrix_rows, train_rows


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys: list[str] = []
    for row in rows:
        for key in row.keys():
            if key not in keys:
                keys.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def _rank_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not rows:
        return []
    ranked = []
    for row in rows:
        r = dict(row)
        style = _safe_float(r.get("all_clip_style"))
        lpips = _safe_float(r.get("all_content_lpips"))
        photo_style = _safe_float(r.get("photo_clip_style"))
        photo_lpips = _safe_float(r.get("photo_content_lpips"))
        r["primary_score"] = _score(style, lpips, style_weight=-1)
        r["weighted_score_65_35"] = _score(style, lpips, style_weight=0.65)
        r["photo_primary_score"] = _score(photo_style, photo_lpips, style_weight=-1)
        ranked.append(r)
    return sorted(ranked, key=lambda x: (x.get("primary_score") is not None, x.get("primary_score") or -999), reverse=True)


def _best_by_experiment(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    best: dict[str, dict[str, Any]] = {}
    for row in _rank_rows(rows):
        best.setdefault(str(row["experiment_id"]), row)
    return _rank_rows(list(best.values()))


def _samst_reference() -> dict[str, Any] | None:
    if not SAMST_SUMMARY.exists():
        return None
    row = _summary_metrics(SAMST_SUMMARY)
    row.update({"experiment_id": "SaMST_strict", "epoch": "strict_750", "recipe_id": "baseline"})
    return row


def _write_report(summary_rows: list[dict[str, Any]], best_rows: list[dict[str, Any]]) -> None:
    report = SWEEP_ROOT / "weight_sweep_40_scientific_conclusions.md"
    samst = _samst_reference()
    top = best_rows[:15]
    lines = [
        "# Weight Sweep 40 Scientific Conclusions",
        "",
        "## Protocol",
        "",
        "- Budget: 40 local experiments = 20 manual category-sampling recipes x `K={1,2}`.",
        "- Base config: `S-add__K-1_C-0_W-20_Col-0/config.json`.",
        "- Training: 8 epochs, checkpoint/evaluation at every epoch.",
        "- Evaluation: strict 750-image protocol, `30 source images x 5 target styles x 5 source styles`, with local CLIP path injected by `run_evaluation.py`.",
        "- Primary scalar score: `EC = CLIP-style * (1 - LPIPS)`. This rewards style strength only when content distortion is not too high.",
        "- Secondary score: `0.65 * CLIP-style + 0.35 * (1 - LPIPS)`. This is less harsh and useful for sanity checking rank stability.",
        "",
    ]
    if not summary_rows:
        lines.extend(["## Status", "", "No evaluated rows are available yet. Run `run_weight_sweep_40.bat` without `--collect_only`."])
    else:
        lines.extend(
            [
                "## Top Runs By Primary Score",
                "",
                "| rank | experiment | epoch | style | content | LPIPS | EC | photo_style | photo_LPIPS | recipe |",
                "|---:|---|---|---:|---:|---:|---:|---:|---:|---|",
            ]
        )
        for idx, row in enumerate(top, 1):
            lines.append(
                f"| {idx} | {row.get('experiment_id')} | {row.get('epoch')} | "
                f"{_safe_float(row.get('all_clip_style')):.4f} | {_safe_float(row.get('all_clip_content')):.4f} | "
                f"{_safe_float(row.get('all_content_lpips')):.4f} | {_safe_float(row.get('primary_score')):.4f} | "
                f"{_safe_float(row.get('photo_clip_style')):.4f} | {_safe_float(row.get('photo_content_lpips')):.4f} | {row.get('recipe_id')} |"
            )
        if samst:
            samst_ec = _score(_safe_float(samst.get("all_clip_style")), _safe_float(samst.get("all_content_lpips")), style_weight=-1)
            lines.extend(
                [
                    "",
                    "## SaMST Reference",
                    "",
                    f"- SaMST strict: style `{_safe_float(samst.get('all_clip_style')):.4f}`, content `{_safe_float(samst.get('all_clip_content')):.4f}`, LPIPS `{_safe_float(samst.get('all_content_lpips')):.4f}`, EC `{samst_ec:.4f}`.",
                ]
            )
            if top:
                best = top[0]
                lines.append(
                    f"- Best sweep row vs SaMST: style delta `{_safe_float(best.get('all_clip_style')) - _safe_float(samst.get('all_clip_style')):+.4f}`, "
                    f"LPIPS delta `{_safe_float(best.get('all_content_lpips')) - _safe_float(samst.get('all_content_lpips')):+.4f}`, "
                    f"EC delta `{_safe_float(best.get('primary_score')) - samst_ec:+.4f}`."
                )
        lines.extend(
            [
                "",
                "## Interpretation Guidance",
                "",
                "- If a run beats SaMST on EC but not raw style, the claim should be content-preserving style transfer rather than stronger raw stylization.",
                "- If a recipe improves `photo_clip_style` but hurts all-pairs EC, it is a candidate for domain-specific inference or category-conditioned weighting, not a global default.",
                "- If K2 recipes dominate EC while K1 dominates style, the next stage should interpolate `K` or add per-target weights instead of changing all losses globally.",
            ]
        )
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")


def collect_outputs(manifest_rows: list[dict[str, str]]) -> None:
    summary_rows, matrix_rows, train_rows = _collect(manifest_rows)
    best_rows = _best_by_experiment(summary_rows)
    _write_csv(SWEEP_ROOT / "weight_sweep_40_all_epochs.csv", summary_rows)
    _write_csv(SWEEP_ROOT / "weight_sweep_40_direction_matrix.csv", matrix_rows)
    _write_csv(SWEEP_ROOT / "weight_sweep_40_train_times.csv", train_rows)
    _write_csv(SWEEP_ROOT / "weight_sweep_40_best_by_experiment.csv", best_rows)
    if summary_rows:
        _write_csv(SWEEP_ROOT / "weight_sweep_40_top_epochs.csv", _rank_rows(summary_rows)[:80])
    _write_report(summary_rows, best_rows)
    print(SWEEP_ROOT / "weight_sweep_40_all_epochs.csv")
    print(SWEEP_ROOT / "weight_sweep_40_best_by_experiment.csv")
    print(SWEEP_ROOT / "weight_sweep_40_scientific_conclusions.md")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run 40 category-weight sweep experiments and evaluate every epoch.")
    parser.add_argument("--start", type=int, default=0, help="Start experiment index in manifest.")
    parser.add_argument("--limit", type=int, default=0, help="Run at most this many experiments; 0 means all.")
    parser.add_argument("--force_train", action="store_true", help="Retrain even when epoch_0008.pt exists.")
    parser.add_argument("--force_eval", action="store_true", help="Re-evaluate even when summaries exist.")
    parser.add_argument("--collect_only", action="store_true", help="Only collect existing summaries into CSV/report.")
    args = parser.parse_args()

    rows = _read_manifest()
    selected = rows[args.start :]
    if args.limit:
        selected = selected[: args.limit]

    status_rows: list[dict[str, Any]] = []
    if not args.collect_only:
        for idx, row in enumerate(selected, args.start):
            print(f"\n=== [{idx + 1}/{len(rows)}] {row['experiment_id']} ===", flush=True)
            status = dict(row)
            status.update(_train_one(row, force_train=args.force_train))
            if int(status.get("train_rc", 1)) == 0:
                status.update(_eval_one(row, force_eval=args.force_eval))
            else:
                status.update({"eval_status": "skipped_train_failed", "eval_rc": "", "eval_time_sec": ""})
            status_rows.append(status)
            _write_csv(SWEEP_ROOT / "weight_sweep_40_status.csv", status_rows)
            collect_outputs(rows)
    else:
        collect_outputs(rows)


if __name__ == "__main__":
    main()
