"""Run evaluation on all completed armored_breakthrough checkpoints."""
import subprocess, sys, os
from pathlib import Path

runs_dir = Path("exp/armored_breakthrough/runs")
for run in sorted(runs_dir.iterdir()):
    if not run.is_dir():
        continue
    ckpt = run / "checkpoints" / "epoch_0007.pt"
    if not ckpt.exists():
        print(f"SKIP {run.name} (no checkpoint)")
        continue
    eval_dir = run / "full_eval" / "epoch_0007"
    summary = eval_dir / "summary.json"
    if summary.exists():
        print(f"SKIP {run.name} (already evaluated)")
        continue
    print(f"Evaluating {run.name}...")
    cmd = [
        sys.executable, "src/utils/run_evaluation.py",
        "--checkpoint", str(ckpt),
        "--output", str(eval_dir),
        "--batch_size", "6",
        "--num_steps", "4",
        "--force_regen",
    ]
    r = subprocess.run(cmd, cwd=str(Path.cwd()))
    if r.returncode != 0:
        print(f"  FAILED: exit {r.returncode}")
    else:
        print(f"  DONE")
print("All done!")
