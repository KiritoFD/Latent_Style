"""Master serial launcher for all run_511 baselines.

Runs training + 750-image inference for each baseline in sequence:
  1. StyTR-2   (self-contained in run_511/repos/StyTR-2)
  2. AdaIN     (self-contained in run_511/repos/adain)
  3. AesFA     (self-contained in run_511/repos/AesFA)
  4. AesPA-Net (self-contained in run_511/repos/AesPA-Net)
  5. StyleID   (training-free, uses diffusers)
  6. SaMST     (self-contained in run_511/repos/SaMST-main)
  7. CAST      (references run_511/repos/cast)

After all baselines complete, optionally runs SB evaluation on all outputs.

Usage:
  python run_511/run_all_511.py                       # full run, all baselines
  python run_511/run_all_511.py --baselines adain aesfa  # only specific baselines
  python run_511/run_all_511.py --mode smoke           # smoke test (1 iter, 1 image)
  python run_511/run_all_511.py --mode infer           # inference only (skip training)
  python run_511/run_all_511.py --mode eval             # evaluation only
"""
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import time
from pathlib import Path


THIS_DIR = Path(__file__).resolve().parent
WORKSPACE_ROOT = THIS_DIR.parent

BASELINES = {
    "stytr2": {
        "script": THIS_DIR / "run_stytr2_750.py",
        "run_root": THIS_DIR / "outputs" / "stytr2_750",
        "label": "StyTR-2",
    },
    "adain": {
        "script": THIS_DIR / "run_adain_750.py",
        "run_root": THIS_DIR / "outputs" / "adain_750",
        "label": "AdaIN",
    },
    "aesfa": {
        "script": THIS_DIR / "run_aesfa_750.py",
        "run_root": THIS_DIR / "outputs" / "aesfa_750",
        "label": "AesFA",
    },
    "aespa": {
        "script": THIS_DIR / "run_aespa_750.py",
        "run_root": THIS_DIR / "outputs" / "aespa_750",
        "label": "AesPA-Net",
    },
    "styleid": {
        "script": THIS_DIR / "run_styleid_750.py",
        "run_root": THIS_DIR / "outputs" / "styleid_750",
        "label": "StyleID",
    },
    "samst": {
        "script": THIS_DIR / "run_samst_750.py",
        "run_root": THIS_DIR / "outputs" / "samst_750",
        "label": "SaMST",
    },
    "cast": {
        "script": THIS_DIR / "run_cast_750.py",
        "run_root": THIS_DIR / "outputs" / "cast_750",
        "label": "CAST",
    },
}


def run_baseline(name: str, cfg: dict, mode: str, profile: str) -> dict[str, object]:
    """Run a single baseline script and return its summary."""
    script = cfg["script"]
    if not script.exists():
        return {"baseline": name, "label": cfg["label"], "status": "missing_script", "script": str(script)}

    cmd = [
        sys.executable, str(script),
        "--mode", mode,
        "--profile", profile,
        "--run_root", str(cfg["run_root"]),
    ]
    print(f"\n{'='*60}")
    print(f"  [{cfg['label']}] mode={mode} profile={profile}")
    print(f"  script: {script}")
    print(f"  output: {cfg['run_root']}")
    print(f"{'='*60}\n")

    start = time.time()
    proc = subprocess.run(cmd, cwd=str(WORKSPACE_ROOT))
    elapsed = round(time.time() - start, 3)

    # Read summary if available
    summary_path = cfg["run_root"] / "summary.json"
    summary = {}
    if summary_path.exists():
        try:
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
        except Exception:
            pass

    return {
        "baseline": name,
        "label": cfg["label"],
        "status": "ok" if proc.returncode == 0 else "failed",
        "returncode": proc.returncode,
        "elapsed_sec": elapsed,
        "mode": mode,
        "profile": profile,
        "summary": summary,
    }


def run_eval(baselines_to_eval: list[str]) -> dict[str, object]:
    """Run SB evaluation on all completed baseline outputs."""
    eval_script = WORKSPACE_ROOT / "SchrodingerBridge" / "src" / "utils" / "run_evaluation.py"
    if not eval_script.exists():
        return {"status": "blocked", "error": f"eval script not found: {eval_script}"}

    # Collect image directories
    image_dirs = []
    for name in baselines_to_eval:
        cfg = BASELINES[name]
        img_dir = cfg["run_root"] / "infer_750" / "images"
        if img_dir.exists() and any(img_dir.glob("*.jpg")):
            image_dirs.append((name, img_dir))

    if not image_dirs:
        return {"status": "blocked", "error": "no baseline outputs found for evaluation"}

    results = []
    for name, img_dir in image_dirs:
        print(f"\n[Evaluating {name}] {img_dir}")
        cmd = [
            sys.executable, str(eval_script),
            "--images_dir", str(img_dir),
            "--output_dir", str(BASELINES[name]["run_root"] / "eval"),
            "--metrics", "clip_style", "clip_content", "content_lpips", "art_fid", "fid",
        ]
        start = time.time()
        proc = subprocess.run(cmd, cwd=str(WORKSPACE_ROOT))
        results.append({
            "baseline": name,
            "status": "ok" if proc.returncode == 0 else "failed",
            "elapsed_sec": round(time.time() - start, 3),
        })

    return {"status": "done", "results": results}


def write_master_summary(rows: list[dict[str, object]], run_root: Path) -> None:
    run_root.mkdir(parents=True, exist_ok=True)
    (run_root / "master_summary.json").write_text(
        json.dumps({"baselines": rows}, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    # Flat CSV for quick comparison
    with (run_root / "master_summary.csv").open("w", encoding="utf-8", newline="") as f:
        flat = []
        for row in rows:
            flat.append({
                "baseline": row.get("baseline", ""),
                "label": row.get("label", ""),
                "status": row.get("status", ""),
                "returncode": row.get("returncode", ""),
                "elapsed_sec": row.get("elapsed_sec", ""),
                "mode": row.get("mode", ""),
            })
        keys = ["baseline", "label", "status", "returncode", "elapsed_sec", "mode"]
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(flat)


def main() -> int:
    parser = argparse.ArgumentParser(description="Master serial launcher for run_511 baselines.")
    parser.add_argument(
        "--baselines",
        nargs="+",
        default=list(BASELINES.keys()),
        choices=list(BASELINES.keys()),
        help="Which baselines to run.",
    )
    parser.add_argument("--mode", choices=["train", "infer", "all", "smoke", "eval"], default="all")
    parser.add_argument("--profile", choices=["4g", "7g", "11g"], default="7g")
    parser.add_argument("--eval_only", action="store_true", help="Skip train/infer, only run eval.")
    args = parser.parse_args()

    output_root = THIS_DIR / "outputs"
    output_root.mkdir(parents=True, exist_ok=True)

    if args.eval_only or args.mode == "eval":
        eval_result = run_eval(args.baselines)
        print(json.dumps(eval_result, indent=2))
        return 0 if eval_result.get("status") == "done" else 1

    rows = []
    start_all = time.time()

    for name in args.baselines:
        cfg = BASELINES[name]
        row = run_baseline(name, cfg, args.mode, args.profile)
        rows.append(row)
        write_master_summary(rows, output_root)

        # Print quick status
        status = row.get("status", "unknown")
        elapsed = row.get("elapsed_sec", 0)
        print(f"\n>>> [{row.get('label', name)}] {status} ({elapsed}s)")

        # Stop on failure for train+infer modes
        if status not in {"ok", "blocked"} and args.mode in {"all", "train"}:
            print(f"\n*** Stopping: {row.get('label', name)} failed ***")
            break

    total_elapsed = round(time.time() - start_all, 3)
    print(f"\n{'='*60}")
    print(f"  Total elapsed: {total_elapsed}s")
    print(f"  Baselines completed: {sum(1 for r in rows if r.get('status') == 'ok')}/{len(rows)}")
    print(f"  Summary: {output_root / 'master_summary.csv'}")
    print(f"{'='*60}")

    return 0 if all(r.get("status") in {"ok", "blocked"} for r in rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
