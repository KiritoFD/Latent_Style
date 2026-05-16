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
OUT_ROOT = ROOT / "theory_switch_validation"
MANIFEST = OUT_ROOT / "manifest.csv"
SAMST_SUMMARY = ROOT.parent / "Related_Works" / "run_511" / "complete_750" / "samst_strict" / "summary.json"
SAMST_SELECTED_CSV = ROOT / "selected_style_metrics.csv"


def _run(cmd: list[str], log_path: Path) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    print(" ".join(str(x) for x in cmd), flush=True)
    with log_path.open("a", encoding="utf-8", errors="replace") as f:
        f.write("\n\n$ " + " ".join(str(x) for x in cmd) + "\n")
        f.flush()
        return subprocess.run(cmd, cwd=ROOT, stdout=f, stderr=subprocess.STDOUT).returncode


def _ensure_manifest() -> list[dict[str, str]]:
    if not MANIFEST.exists():
        rc = subprocess.run([sys.executable, str(ROOT / "prepare_theory_switch_validation.py")], cwd=ROOT).returncode
        if rc != 0:
            raise SystemExit(rc)
    with MANIFEST.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _f(value: Any) -> float | None:
    try:
        if value is None or value == "":
            return None
        return float(value)
    except Exception:
        return None


def _ec(style: Any, lpips: Any) -> float | None:
    s = _f(style)
    l = _f(lpips)
    if s is None or l is None:
        return None
    return s * (1.0 - l)


def _summary_metrics(path: Path) -> dict[str, Any]:
    data = _load_json(path)
    analysis = data.get("analysis", {}) or {}
    all_pairs = analysis.get("all_pairs_overview", {}) or {}
    transfer = analysis.get("style_transfer_ability", {}) or {}
    photo = analysis.get("photo_to_art_performance", {}) or {}
    row = {
        "summary_path": str(path),
        "checkpoint": data.get("checkpoint", ""),
        "all_clip_style": all_pairs.get("clip_style"),
        "all_clip_content": all_pairs.get("clip_content"),
        "all_content_lpips": all_pairs.get("content_lpips"),
        "transfer_clip_style": transfer.get("clip_style"),
        "transfer_clip_content": transfer.get("clip_content"),
        "transfer_content_lpips": transfer.get("content_lpips"),
        "photo_clip_style": photo.get("clip_style"),
        "photo_clip_content": photo.get("clip_content"),
        "photo_content_lpips": photo.get("content_lpips"),
    }
    row["ec_all"] = _ec(row["all_clip_style"], row["all_content_lpips"])
    row["ec_transfer"] = _ec(row["transfer_clip_style"], row["transfer_content_lpips"])
    row["ec_photo"] = _ec(row["photo_clip_style"], row["photo_content_lpips"])
    return row


def _samst_reference() -> dict[str, Any] | None:
    if SAMST_SUMMARY.exists():
        return _summary_metrics(SAMST_SUMMARY)
    if SAMST_SELECTED_CSV.exists():
        with SAMST_SELECTED_CSV.open("r", encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f):
                if str(row.get("method", "")).lower() == "samst":
                    out = {
                        "all_clip_style": row.get("clip_style_up"),
                        "all_clip_content": row.get("clip_content_up"),
                        "all_content_lpips": row.get("lpips_down"),
                        "photo_clip_style": "",
                        "photo_clip_content": "",
                        "photo_content_lpips": "",
                    }
                    out["ec_all"] = _ec(out["all_clip_style"], out["all_content_lpips"])
                    return out
    return None


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _train(row: dict[str, str], force: bool) -> dict[str, Any]:
    run_dir = Path(row["run_dir"])
    if (run_dir / "epoch_0003.pt").exists() and not force:
        return {"train_status": "skipped_existing", "train_rc": 0, "train_sec": ""}
    start = time.time()
    rc = _run(
        [sys.executable, "run.py", "--config", row["config_path"]],
        OUT_ROOT / "logs" / f"{row['experiment_id']}.train.log",
    )
    return {"train_status": "ok" if rc == 0 else "failed", "train_rc": rc, "train_sec": round(time.time() - start, 3)}


def _eval(row: dict[str, str], force: bool) -> dict[str, Any]:
    run_dir = Path(row["run_dir"])
    output_root = run_dir / "full_eval"
    if (output_root / "batch_summary.csv").exists() and not force:
        return {"eval_status": "skipped_existing", "eval_rc": 0, "eval_sec": ""}
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
    if force:
        cmd.append("--force")
    start = time.time()
    rc = _run(cmd, OUT_ROOT / "logs" / f"{row['experiment_id']}.eval.log")
    return {"eval_status": "ok" if rc == 0 else "failed", "eval_rc": rc, "eval_sec": round(time.time() - start, 3)}


def collect(rows: list[dict[str, str]]) -> None:
    all_rows: list[dict[str, Any]] = []
    for row in rows:
        run_dir = Path(row["run_dir"])
        for summary in sorted((run_dir / "full_eval").glob("epoch_*/summary.json")):
            epoch = summary.parent.name
            out = dict(row)
            out.update({"epoch": epoch, "epoch_num": int(epoch.split("_")[-1])})
            out.update(_summary_metrics(summary))
            all_rows.append(out)
    ranked = sorted(all_rows, key=lambda r: (_f(r.get("ec_all")) is not None, _f(r.get("ec_all")) or -999), reverse=True)
    best_by_exp: dict[str, dict[str, Any]] = {}
    for row in ranked:
        best_by_exp.setdefault(str(row["experiment_id"]), row)
    best_rows = sorted(best_by_exp.values(), key=lambda r: (_f(r.get("ec_all")) is not None, _f(r.get("ec_all")) or -999), reverse=True)
    _write_csv(OUT_ROOT / "theory_switch_validation_all_epochs.csv", all_rows)
    _write_csv(OUT_ROOT / "theory_switch_validation_best_by_experiment.csv", best_rows)
    _write_report(all_rows, best_rows)


def _write_report(all_rows: list[dict[str, Any]], best_rows: list[dict[str, Any]]) -> None:
    samst = _samst_reference()
    lines = [
        "# Theory Switch Validation",
        "",
        "## Design",
        "",
        "- Goal: verify whether the new optional switches improve the style/content Pareto point, not to replace the main model blindly.",
        "- Base: original `S-add__K-1_C-0_W-20_Col-0/config.json`, with `K=2`, `terminal_swd_weight=20`, `w_cycle=0`, 3 epochs.",
        "- Evaluation: every epoch on the strict 750-image protocol.",
        "- Primary score: `EC = CLIP-style * (1 - LPIPS)`.",
        "",
    ]
    if not best_rows:
        lines.append("No completed evaluation rows yet.")
    else:
        lines.extend(["## Best Epoch Per Variant", "", "| rank | variant | epoch | style | content | LPIPS | EC | photo_style | photo_LPIPS | note |", "|---:|---|---|---:|---:|---:|---:|---:|---:|---|"])
        for idx, row in enumerate(best_rows, 1):
            lines.append(
                f"| {idx} | {row['experiment_id']} | {row['epoch']} | {_f(row['all_clip_style']):.4f} | {_f(row['all_clip_content']):.4f} | "
                f"{_f(row['all_content_lpips']):.4f} | {_f(row['ec_all']):.4f} | {_f(row['photo_clip_style']):.4f} | {_f(row['photo_content_lpips']):.4f} | {row['note']} |"
            )
        baseline = next((r for r in best_rows if r["experiment_id"] == "T0_k2_baseline"), None)
        if baseline:
            lines.extend(["", "## Delta Against T0 Baseline", "", "| variant | Delta style | Delta content | Delta LPIPS | Delta EC | reading |", "|---|---:|---:|---:|---:|---|"])
            for row in best_rows:
                if row["experiment_id"] == "T0_k2_baseline":
                    continue
                ds = _f(row["all_clip_style"]) - _f(baseline["all_clip_style"])
                dc = _f(row["all_clip_content"]) - _f(baseline["all_clip_content"])
                dl = _f(row["all_content_lpips"]) - _f(baseline["all_content_lpips"])
                de = _f(row["ec_all"]) - _f(baseline["ec_all"])
                reading = "promising" if de > 0.002 else ("mixed" if abs(de) <= 0.002 else "negative")
                lines.append(f"| {row['experiment_id']} | {ds:+.4f} | {dc:+.4f} | {dl:+.4f} | {de:+.4f} | {reading} |")
        if samst:
            lines.extend(
                [
                    "",
                    "## SaMST Reference",
                    "",
                    f"- SaMST strict: style `{_f(samst['all_clip_style']):.4f}`, content `{_f(samst['all_clip_content']):.4f}`, LPIPS `{_f(samst['all_content_lpips']):.4f}`, EC `{_f(samst['ec_all']):.4f}`.",
                ]
            )
            top = best_rows[0]
            lines.append(
                f"- Best validation row vs SaMST: Delta style `{_f(top['all_clip_style']) - _f(samst['all_clip_style']):+.4f}`, "
                f"Delta LPIPS `{_f(top['all_content_lpips']) - _f(samst['all_content_lpips']):+.4f}`, Delta EC `{_f(top['ec_all']) - _f(samst['ec_all']):+.4f}`."
            )
    (OUT_ROOT / "theory_switch_validation_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--collect_only", action="store_true")
    parser.add_argument("--force_train", action="store_true")
    parser.add_argument("--force_eval", action="store_true")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()
    rows = _ensure_manifest()
    selected = rows[args.start :]
    if args.limit:
        selected = selected[: args.limit]

    status_rows: list[dict[str, Any]] = []
    if not args.collect_only:
        for idx, row in enumerate(selected, args.start + 1):
            print(f"\n=== [{idx}/{len(rows)}] {row['experiment_id']} ===", flush=True)
            status = dict(row)
            status.update(_train(row, args.force_train))
            train_rc = status.get("train_rc")
            if train_rc == "" or train_rc is None:
                train_rc = 1
            if int(train_rc) == 0:
                status.update(_eval(row, args.force_eval))
            else:
                status.update({"eval_status": "skipped_train_failed", "eval_rc": "", "eval_sec": ""})
            status_rows.append(status)
            _write_csv(OUT_ROOT / "theory_switch_validation_status.csv", status_rows)
            collect(rows)
    else:
        collect(rows)
    print(OUT_ROOT / "theory_switch_validation_best_by_experiment.csv")
    print(OUT_ROOT / "theory_switch_validation_report.md")


if __name__ == "__main__":
    main()
