"""Chain runner for D3/D4/D5 matched ablations.

Waits for batch 1 chain runner to complete, then trains and evaluates D3/D4/D5.
Run this as a background process:
    python -u scripts/chain_D3_D4_D5.py > logs/chain_D3_D4_D5.log 2>&1
"""
import subprocess
import sys
import time
from pathlib import Path

WEAVE_ROOT = Path(r"I:\Github\Latent_Style\WEAVE")
LOG_DIR = Path(r"C:\Users\Administrator\logs")
BATCH1_LOG = LOG_DIR / "chain_after_seed7.log"
BATCH1_EXIT_MARKER = "CHAIN_RUNNER_EXIT=0"

CONFIGS = [
    ("D3", "configs/rebuttal_D3_wll_1p0.json", "runs/submission/rebuttal_D3_wll_1p0"),
    ("D4", "configs/rebuttal_D4_direct_target.json", "runs/submission/rebuttal_D4_direct_target"),
    ("D5", "configs/rebuttal_D5_hh_head.json", "runs/submission/rebuttal_D5_hh_head"),
]


def log(msg):
    print(msg, flush=True)
    with open(LOG_DIR / "chain_D3_D4_D5.log", "a", encoding="utf-8") as f:
        f.write(msg + "\n")


def wait_for_batch1():
    log(f"=== CHAIN D3/D4/D5: waiting for batch1 @ {time.strftime('%Y-%m-%d %H:%M:%S')} ===")
    while True:
        if BATCH1_LOG.exists():
            content = BATCH1_LOG.read_text(encoding="utf-8", errors="replace")
            if BATCH1_EXIT_MARKER in content:
                log(f"Batch1 done @ {time.strftime('%Y-%m-%d %H:%M:%S')}")
                return True
        log(f"Waiting for batch1... {time.strftime('%Y-%m-%d %H:%M:%S')}")
        time.sleep(120)


def run_cmd(cmd, log_name, cwd=WEAVE_ROOT):
    log_path = LOG_DIR / log_name
    log(f"  Running: {' '.join(cmd)} -> {log_path}")
    with open(log_path, "w", encoding="utf-8") as f:
        proc = subprocess.run(cmd, cwd=str(cwd), stdout=f, stderr=subprocess.STDOUT)
    log(f"  Exit code: {proc.returncode}")
    return proc.returncode


def main():
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    # Step 1: Wait for batch 1
    wait_for_batch1()

    # Step 2: D3/D4/D5 train + eval
    for tag, config_path, run_dir in CONFIGS:
        log(f"\n=== {tag}: Training ({config_path}) @ {time.strftime('%Y-%m-%d %H:%M:%S')} ===")
        train_cmd = [sys.executable, "-u", "run.py", "--config", config_path]
        run_cmd(train_cmd, f"rebuttal_{tag}_train.log")

        log(f"=== {tag}: Per-epoch evaluation @ {time.strftime('%Y-%m-%d %H:%M:%S')} ===")
        eval_cmd = [
            sys.executable, "-u", "scripts/expA_per_epoch_eval.py",
            "--run_dir", run_dir,
            "--seed", "42",
            "--tag", tag,
        ]
        run_cmd(eval_cmd, f"rebuttal_{tag}_eval.log")

    log(f"\n=== CHAIN_D3_D4_D5 EXIT=0 @ {time.strftime('%Y-%m-%d %H:%M:%S')} ===")


if __name__ == "__main__":
    main()
