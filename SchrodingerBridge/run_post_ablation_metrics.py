from __future__ import annotations

import argparse
import csv
import subprocess
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
WORKSPACE_ROOT = ROOT.parent
ABLATION_ROOT = ROOT / "ablation_destructive_7epoch"
RUN511_EVAL = WORKSPACE_ROOT / "Related_Works" / "run_511" / "eval"
METRICS_DIR = WORKSPACE_ROOT / "Related_Works" / "results" / "metrics_summary"


TASKS = {
    "base": ("eval_750.py", "eval_protocol750_sbmatch.json"),
    "guard": ("eval_guard_750.py", "eval_guard750.json"),
    "artifact": ("eval_artifact_pack_750.py", "eval_artifact_pack750.json"),
    "hf_kid": ("eval_hf_patch_kid_750.py", "eval_hf_patch_kid750.json"),
    "plain_kid": ("eval_plain_kid_750.py", "eval_plain_kid750.json"),
}


def read_registry() -> list[dict[str, str]]:
    path = ABLATION_ROOT / "destructive_ablation_7epoch_registry.csv"
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def image_dir_for(row: dict[str, str]) -> Path:
    run_dir = Path(row["save_dir"])
    return run_dir / "full_eval" / "epoch_0007" / "images"


def run_eval_task(task: str, images_dir: Path, *, force: bool, max_ref_cache: int) -> dict[str, Any]:
    script, out_name = TASKS[task]
    output = images_dir.parent / out_name
    if output.exists() and not force:
        return {"task": task, "status": "skipped_existing", "output": str(output)}
    cmd = [
        sys.executable,
        str(RUN511_EVAL / script),
        "--images_dir",
        str(images_dir),
        "--output",
        str(output),
    ]
    if task in {"base", "guard", "artifact"}:
        cmd.extend(["--max_ref_cache", str(max_ref_cache)])
    proc = subprocess.run(cmd, cwd=str(WORKSPACE_ROOT))
    return {
        "task": task,
        "status": "ok" if proc.returncode == 0 and output.exists() else f"failed:{proc.returncode}",
        "output": str(output),
    }


def write_status(rows: list[dict[str, Any]]) -> None:
    ABLATION_ROOT.mkdir(parents=True, exist_ok=True)
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    keys = ["id", "task", "status", "output"]
    for path in [
        ABLATION_ROOT / "destructive_ablation_advanced_metrics_status.csv",
        METRICS_DIR / "destructive_ablation_advanced_metrics_status.csv",
    ]:
        with path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows([{k: row.get(k, "") for k in keys} for row in rows])


def main() -> int:
    parser = argparse.ArgumentParser(description="Run run_511 metric pack on completed 7-epoch ablations.")
    parser.add_argument("--only", nargs="*", default=[], help="Optional ablation IDs.")
    parser.add_argument(
        "--tasks",
        nargs="*",
        default=["base", "guard"],
        choices=sorted(TASKS),
        help="Metric tasks to run. Use artifact/hf_kid/plain_kid after the base table is stable.",
    )
    parser.add_argument("--max_ref_cache", type=int, default=64)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    wanted = set(args.only)
    status_rows: list[dict[str, Any]] = []
    for row in read_registry():
        run_id = row.get("id", "")
        if wanted and run_id not in wanted:
            continue
        images_dir = image_dir_for(row)
        if not images_dir.exists():
            for task in args.tasks:
                status_rows.append({"id": run_id, "task": task, "status": "missing_images", "output": str(images_dir)})
            continue
        for task in args.tasks:
            result = run_eval_task(task, images_dir, force=args.force, max_ref_cache=args.max_ref_cache)
            result["id"] = run_id
            status_rows.append(result)
            write_status(status_rows)
    write_status(status_rows)
    print(ABLATION_ROOT / "destructive_ablation_advanced_metrics_status.csv")
    print(METRICS_DIR / "destructive_ablation_advanced_metrics_status.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
