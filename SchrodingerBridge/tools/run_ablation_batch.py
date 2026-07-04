#!/usr/bin/env python3
"""
Batch train + eval+WFI for 620 ablation experiments.

Reads a JSON list of experiment specs and runs each one sequentially:
1. Train with `python src/run.py --config <config_path>`
2. Run eval+WFI with `tools/run_eval_with_wfi.py`
3. Collect metrics into a unified CSV/JSON summary.

Experiment spec format:
[
  {
    "name": "620_ablation_attn_softmax_smoke",
    "config": "configs/ablations/620_ablation_attn_softmax_smoke.json",
    "epochs": ["epoch_0001"]
  },
  ...
]

Usage:
  python tools/run_ablation_batch.py \
    --spec configs/ablations/ablation_batch.json \
    --test-dir F:/wikiart_distinct5_samam_512_classview/test \
    --cache-dir F:/eval_cache \
    --source-dir F:/wikiart_distinct5_samam_512_classview/test \
    --out results/ablation_summary.csv
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
RUN_PY = REPO_ROOT / "src" / "run.py"
EVAL_PY = REPO_ROOT / "tools" / "run_eval_with_wfi.py"

LOCAL_DEFAULTS = {
    "test_dir": "F:/wikiart_distinct5_samam_512_classview/test",
    "cache_dir": "F:/eval_cache",
    "clip_hf_cache_dir": "F:/eval_cache/hf",
    "source_dir": "F:/wikiart_distinct5_samam_512_classview/test",
}


def load_spec(path: str) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict):
        return data.get("experiments", [data])
    return data


def find_checkpoint(exp_dir: Path, epoch_name: str) -> Path | None:
    candidates = [
        exp_dir / f"{epoch_name}.pt",
        exp_dir / "checkpoints" / f"{epoch_name}.pt",
    ]
    for c in candidates:
        if c.exists():
            return c
    return None


def run_train(config_path: str, verbose: bool = False) -> tuple[bool, float]:
    cmd = [sys.executable, str(RUN_PY), "--config", config_path]
    print(f"\n[TRAIN] {' '.join(cmd)}")
    start = time.time()
    result = subprocess.run(cmd, capture_output=not verbose, text=True)
    elapsed = time.time() - start
    if result.returncode != 0:
        print(f"[TRAIN FAILED] exit={result.returncode}, time={elapsed:.1f}s")
        if not verbose:
            print(result.stderr[-2000:] if result.stderr else "(empty stderr)")
        return False, elapsed
    print(f"[TRAIN OK] time={elapsed:.1f}s")
    return True, elapsed


def run_eval(
    checkpoint: Path,
    output_dir: Path,
    args: argparse.Namespace,
    verbose: bool = False,
) -> dict | None:
    cmd = [
        sys.executable,
        str(EVAL_PY),
        "--checkpoint", str(checkpoint),
        "--output", str(output_dir),
        "--test-dir", args.test_dir,
        "--cache-dir", args.cache_dir,
        "--clip-hf-cache-dir", args.clip_hf_cache_dir,
        "--source-dir", args.source_dir,
        "--batch-size", str(args.batch_size),
        "--target-chunk-size", str(args.target_chunk_size),
        "--vae-decode-batch-size", str(args.vae_decode_batch_size),
        "--eval-lpips-chunk-size", str(args.eval_lpips_chunk_size),
        "--num-steps", str(args.num_steps),
    ]
    if args.clip_style_idt_baseline is not None:
        cmd += ["--clip-style-idt-baseline", str(args.clip_style_idt_baseline)]
    if verbose:
        cmd.append("--verbose")

    print(f"\n[EVAL] {' '.join(cmd)}")
    start = time.time()
    result = subprocess.run(cmd, capture_output=not verbose, text=True)
    elapsed = time.time() - start
    if result.returncode != 0:
        print(f"[EVAL FAILED] exit={result.returncode}, time={elapsed:.1f}s")
        if not verbose:
            print(result.stderr[-2000:] if result.stderr else "(empty stderr)")
        return None

    report_path = output_dir / "wfi_eval_report.json"
    if not report_path.exists():
        print(f"[EVAL WARNING] report not found: {report_path}")
        return None
    with open(report_path, "r", encoding="utf-8") as f:
        report = json.load(f)
    report["eval_wall_sec"] = elapsed
    return report


def run_experiment(spec: dict, args: argparse.Namespace) -> dict:
    name = spec["name"]
    config_path = spec["config"]
    epochs = spec.get("epochs", ["epoch_0001"])
    skip_train = spec.get("skip_train", False)

    print(f"\n{'='*70}")
    print(f"EXPERIMENT: {name}")
    print(f"CONFIG: {config_path}")
    print(f"{'='*70}")

    record = {
        "name": name,
        "config": config_path,
        "status": "pending",
        "train_wall_sec": 0.0,
        "epochs": {},
    }

    if not Path(config_path).is_absolute():
        config_path = str(REPO_ROOT / config_path)

    # Training
    if not skip_train:
        ok, train_sec = run_train(config_path, verbose=args.verbose)
        record["train_wall_sec"] = train_sec
        if not ok:
            record["status"] = "train_failed"
            return record
    else:
        print("[TRAIN] skipped (--skip-train or spec)")

    # Resolve experiment save dir from config
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    save_dir = cfg.get("checkpoint", {}).get("save_dir", "")
    if not save_dir:
        save_dir = f"./exp/620_spatial_bridge/{name}"
    exp_dir = Path(save_dir).resolve()

    # Evaluation for each requested epoch
    all_ok = True
    for epoch in epochs:
        ckpt = find_checkpoint(exp_dir, epoch)
        if ckpt is None:
            print(f"[EVAL SKIP] checkpoint not found: {exp_dir}/{epoch}.pt")
            record["epochs"][epoch] = {"status": "checkpoint_missing"}
            all_ok = False
            continue

        eval_dir = exp_dir / "full_eval_wfi" / epoch
        report = run_eval(ckpt, eval_dir, args, verbose=args.verbose)
        if report is None:
            record["epochs"][epoch] = {"status": "eval_failed"}
            all_ok = False
        else:
            record["epochs"][epoch] = {
                "status": "ok",
                **report,
            }

    record["status"] = "ok" if all_ok else "partial_failed"
    return record


def flatten_records(records: list[dict]) -> list[dict]:
    rows = []
    for rec in records:
        base = {
            "name": rec["name"],
            "config": rec["config"],
            "status": rec["status"],
            "train_wall_sec": rec.get("train_wall_sec", 0.0),
        }
        for epoch, metrics in rec.get("epochs", {}).items():
            row = dict(base)
            row["epoch"] = epoch
            row["epoch_status"] = metrics.get("status", "unknown")
            if metrics.get("status") == "ok":
                for k, v in metrics.items():
                    if k != "status":
                        row[k] = v
            rows.append(row)
    return rows


def write_summary(records: list[dict], out_csv: str, out_json: str | None) -> None:
    rows = flatten_records(records)

    # CSV
    if rows:
        fieldnames = list(rows[0].keys())
    else:
        fieldnames = ["name", "config", "status", "train_wall_sec", "epoch", "epoch_status"]

    os.makedirs(Path(out_csv).parent, exist_ok=True)
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    print(f"[SUMMARY CSV] {out_csv}")

    # JSON
    if out_json:
        os.makedirs(Path(out_json).parent, exist_ok=True)
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(records, f, indent=2, ensure_ascii=False)
        print(f"[SUMMARY JSON] {out_json}")


def main():
    parser = argparse.ArgumentParser(description="Batch train+eval+WFI for ablations")
    parser.add_argument("--spec", required=True, help="JSON experiment spec")
    parser.add_argument("--test-dir", default=LOCAL_DEFAULTS["test_dir"])
    parser.add_argument("--cache-dir", default=LOCAL_DEFAULTS["cache_dir"])
    parser.add_argument("--clip-hf-cache-dir", default=LOCAL_DEFAULTS["clip_hf_cache_dir"])
    parser.add_argument("--source-dir", default=LOCAL_DEFAULTS["source_dir"])
    parser.add_argument("--batch-size", default=4, type=int)
    parser.add_argument("--target-chunk-size", default=2, type=int)
    parser.add_argument("--vae-decode-batch-size", default=8, type=int)
    parser.add_argument("--eval-lpips-chunk-size", default=4, type=int)
    parser.add_argument("--num-steps", default=8, type=int)
    parser.add_argument("--clip-style-idt-baseline", default=None, type=float)
    parser.add_argument("--out", default="results/ablation_summary.csv")
    parser.add_argument("--out-json", default="results/ablation_summary.json")
    parser.add_argument("--skip-train", action="store_true", help="Only run eval on existing checkpoints")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    specs = load_spec(args.spec)
    print(f"=== Batch Ablation: {len(specs)} experiments ===")
    for s in specs:
        print(f"  - {s['name']}: {s['config']}")

    if args.dry_run:
        print("\n(dry run, exiting)")
        return

    records = []
    for spec in specs:
        if args.skip_train:
            spec["skip_train"] = True
        rec = run_experiment(spec, args)
        records.append(rec)

    write_summary(records, args.out, args.out_json)

    # Print quick table
    print("\n" + "="*90)
    print(f"{'Experiment':<40} {'Epoch':<12} {'Status':<12} {'WFI':>8} {'Clip-S':>8} {'LPIPS':>8}")
    print("-"*90)
    for rec in records:
        for epoch, m in rec.get("epochs", {}).items():
            status = m.get("status", "unknown")
            if status == "ok":
                print(f"{rec['name']:<40} {epoch:<12} {status:<12} "
                      f"{m.get('wfi_score', 0):>8.4f} {m.get('clip_style', 0):>8.4f} {m.get('content_lpips', 0):>8.4f}")
            else:
                print(f"{rec['name']:<40} {epoch:<12} {status:<12}")


if __name__ == "__main__":
    main()
