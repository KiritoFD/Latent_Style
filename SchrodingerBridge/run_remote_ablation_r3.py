"""Remote ablation runner — Round 3 (triple-axis combinations + sweeps).

Runs 6 Round-3 experiments sequentially on the remote RTX 3060.
Base: r2_spectral_10ep (Round 2 best, CLIP-S=0.7288).

Round 3 combines the three independent improvement axes found in Round 2:
  1. Train length: 5ep -> 10ep (strict win, +0.0051 CLIP-S)
  2. LL weight: 0.3 -> 1.0 (strict win on content at 5ep)
  3. SWD weight: 12 -> 6 (near strict win: style -0.0011, content -0.0159)

Order: fastest first for early signal.
  r3_swd6_llw1_5ep   (5ep,  ~20min) - fast early signal, isolate 10ep effect
  r3_swd6_llw1_10ep  (10ep, ~40min) - TRIPLE COMBO, expected best
  r3_swd9_llw1_10ep  (10ep, ~40min) - swd_w gap fill (6-12)
  r3_llw05_10ep      (10ep, ~40min) - ll_w middle point
  r3_llw2_10ep       (10ep, ~40min) - ll_w upper probe
  r3_spectral_15ep   (15ep, ~60min) - 10->15ep continuation check

Usage (on remote):
    python run_remote_ablation_r3.py
"""
import json
import os
import subprocess
import sys
import time
from datetime import datetime

REPO = r"I:\Github\Latent_Style\SchrodingerBridge"
PYTHON = r"C:\Program Files\Python312\python.exe"
LOG = os.path.join(REPO, "remote_ablation_r3_log.txt")
PID = os.path.join(REPO, "remote_ablation_r3_runner.pid")

# Round 3: 6 experiments. Ordered fastest-first for early signal.
EXPERIMENTS = [
    # (name, config, description, group)
    ("r3_swd6_llw1_5ep",   "configs/semantic_swd_musiq/r3_swd6_llw1_5ep.json",   "swd_w=6 + ll_w=1.0 at 5ep (isolate 10ep effect)",  1),
    ("r3_swd6_llw1_10ep",  "configs/semantic_swd_musiq/r3_swd6_llw1_10ep.json",  "TRIPLE COMBO: swd_w=6 + ll_w=1.0 + 10ep",          2),
    ("r3_swd9_llw1_10ep",  "configs/semantic_swd_musiq/r3_swd9_llw1_10ep.json",  "swd_w=9 + ll_w=1.0 + 10ep (gap fill 6-12)",        2),
    ("r3_llw05_10ep",      "configs/semantic_swd_musiq/r3_llw05_10ep.json",      "ll_w=0.5 + 10ep (ll_w middle point)",              3),
    ("r3_llw2_10ep",       "configs/semantic_swd_musiq/r3_llw2_10ep.json",       "ll_w=2.0 + 10ep (ll_w upper probe)",               3),
    ("r3_spectral_15ep",   "configs/semantic_swd_musiq/r3_spectral_15ep.json",   "spectral default at 15ep (10->15 continuation)",   4),
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
        log(f"=== Round 3 queue start: {len(EXPERIMENTS)} experiments ===")
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
        log(f"=== Round 3 queue done: {ok} ok, {fail} fail, {skip} skip ===")
        return 0 if fail == 0 else 2
    finally:
        if os.path.exists(PID):
            os.remove(PID)


if __name__ == "__main__":
    sys.exit(main())
