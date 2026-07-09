"""Remote ablation runner — Round 2 (spectral-based combinations + sweeps).

Runs 6 Round-2 experiments sequentially on the remote RTX 3060.
Base: spectral (Round 1 best, CLIP-S=0.7237).
- R2-1: spectral + no_swd / + ll_w1 / + both (5ep, 3 configs)
- R2-2: spectral + swd_weight sweep {3.0, 6.0} (5ep, 2 configs)
- R2-4: spectral at 10ep (1 config)

Usage (on remote):
    python run_remote_ablation_r2.py
"""
import json
import os
import subprocess
import sys
import time
from datetime import datetime

REPO = r"I:\Github\Latent_Style\SchrodingerBridge"
PYTHON = r"C:\Program Files\Python312\python.exe"
LOG = os.path.join(REPO, "remote_ablation_r2_log.txt")
PID = os.path.join(REPO, "remote_ablation_r2_runner.pid")

# Round 2: 6 experiments based on spectral (Round 1 best).
EXPERIMENTS = [
    # (name, config, description, group)
    ("r2_spec_noswd",       "configs/semantic_swd_musiq/r2_spec_noswd.json",        "spectral + no_swd_loss (swd_w=0)",         1),
    ("r2_spec_llw1",        "configs/semantic_swd_musiq/r2_spec_llw1.json",         "spectral + spectral_w_ll=1.0",             1),
    ("r2_spec_noswd_llw1",  "configs/semantic_swd_musiq/r2_spec_noswd_llw1.json",   "spectral + swd_w=0 + ll_w=1.0 (triple)",   1),
    ("r2_spec_swd3",        "configs/semantic_swd_musiq/r2_spec_swd3.json",         "spectral + swd_weight=3.0 (0.25x)",        2),
    ("r2_spec_swd6",        "configs/semantic_swd_musiq/r2_spec_swd6.json",         "spectral + swd_weight=6.0 (0.5x)",         2),
    ("r2_spectral_10ep",    "configs/semantic_swd_musiq/r2_spectral_10ep.json",     "spectral at 10 epochs (longer training)",  4),
]


def log(msg: str) -> None:
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def is_complete(name: str) -> bool:
    """Check if experiment has a valid summary.json with CLIP-S."""
    exp_dir = os.path.join(REPO, "exp", name, "full_eval")
    if not os.path.isdir(exp_dir):
        return False
    for epoch_dir in os.listdir(exp_dir):
        summary = os.path.join(exp_dir, epoch_dir, "summary.json")
        if not os.path.exists(summary):
            continue
        try:
            with open(summary, encoding="utf-8") as f:
                data = json.load(f)
            apo = data.get("analysis", {}).get("all_pairs_overview", {})
            if apo.get("clip_style") is not None:
                return True
        except Exception:
            pass
    return False


def run_one(name: str, config: str, desc: str, group: int) -> bool:
    log(f"=== START [G{group}] {name}: {desc} ===")
    start = time.time()
    exp_log = os.path.join(REPO, f"exp_log_{name}.txt")
    try:
        with open(exp_log, "w", encoding="utf-8") as stdout_f:
            result = subprocess.run(
                [PYTHON, "src/run.py", "--config", config],
                cwd=REPO,
                stdout=stdout_f,
                stderr=subprocess.PIPE,
                text=True,
                timeout=7200,
                env=os.environ,
            )
        elapsed = time.time() - start
        if result.returncode == 0:
            log(f"=== DONE  [G{group}] {name}: SUCCESS ({elapsed/60:.1f} min) ===")
            return True
        log(f"=== DONE  [G{group}] {name}: FAILED rc={result.returncode} ({elapsed/60:.1f} min) ===")
        stderr_tail = (result.stderr or "")[-2000:]
        if stderr_tail:
            log(f"  STDERR_TAIL: {stderr_tail}")
        try:
            with open(exp_log, encoding="utf-8") as f:
                lines = f.readlines()
            stdout_tail = "".join(lines[-40:])
            if stdout_tail:
                log(f"  STDOUT_TAIL: {stdout_tail}")
        except Exception:
            pass
        return False
    except subprocess.TimeoutExpired:
        elapsed = time.time() - start
        log(f"=== DONE  [G{group}] {name}: TIMEOUT ({elapsed/60:.1f} min) ===")
        return False
    except Exception as e:
        elapsed = time.time() - start
        log(f"=== DONE  [G{group}] {name}: ERROR {e} ({elapsed/60:.1f} min) ===")
        return False


def main() -> int:
    if os.path.exists(PID):
        with open(PID, encoding="utf-8") as f:
            old = f.read().strip()
        log(f"FATAL: runner already active (PID={old}). Delete {PID} to force.")
        return 1
    with open(PID, "w", encoding="utf-8") as f:
        f.write(str(os.getpid()))

    try:
        log(f"=== Round 2 queue start: {len(EXPERIMENTS)} experiments ===")
        log(f"CUDA_ALLOC_CONF: {os.environ.get('PYTORCH_CUDA_ALLOC_CONF', 'default')}")
        ok = 0
        fail = 0
        skip = 0
        for name, config, desc, group in EXPERIMENTS:
            if is_complete(name):
                log(f"--- SKIP  [G{group}] {name}: already complete ---")
                skip += 1
                continue
            if run_one(name, config, desc, group):
                ok += 1
            else:
                fail += 1
            time.sleep(5)
        log(f"=== Round 2 queue done: {ok} ok, {fail} fail, {skip} skip ===")
        return 0 if fail == 0 else 2
    finally:
        if os.path.exists(PID):
            os.remove(PID)


if __name__ == "__main__":
    sys.exit(main())
