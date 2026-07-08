"""Remote ablation runner — clean single-script infra.

Runs all 16 WEAVE ablation experiments sequentially on the remote RTX 3060.
- bs=112 (empirically safe: 10.4GB VRAM across all configs incl. sinkhorn/spectral)
- PID lock prevents duplicate runs
- Auto-skips experiments with valid summary.json
- Per-experiment logs to exp_log_{name}.txt
- 2h per-experiment timeout

Usage (on remote):
    python run_remote_ablation.py
"""
import json
import os
import subprocess
import sys
import time
from datetime import datetime

REPO = r"I:\Github\Latent_Style\SchrodingerBridge"
PYTHON = r"C:\Program Files\Python312\python.exe"
LOG = os.path.join(REPO, "remote_ablation_log.txt")
PID = os.path.join(REPO, "remote_ablation_runner.pid")

# 16 experiments covering all WEAVE components.
# Tier 1: core component removal (4)  — run first, largest signal
# Tier 2: Semantic SWD mechanism (4)  — blend/K/mode ablations
# Tier 3: architecture extremes (4)   — LL weight, route prob
# Tier 4: mechanism replacements (4)  — sinkhorn/spectral/soft/attn
EXPERIMENTS = [
    # (name, config, description, tier)
    ("abl_baseline",    "configs/semantic_swd_musiq/abl_baseline.json",         "Full WEAVE baseline",            0),
    # Tier 1 — core component removal
    ("abl_no_swd_loss", "configs/semantic_swd_musiq/abl_no_swd_loss.json",      "Remove SWD loss",                1),
    ("abl_no_dwt_route","configs/semantic_swd_musiq/abl_no_dwt_route.json",     "Remove DWT high-freq routing",   1),
    ("abl_no_wct",      "configs/semantic_swd_musiq/abl_no_wct.json",           "Remove Endpoint WCT",            1),
    ("abl_no_eota",     "configs/semantic_swd_musiq/abl_no_eota.json",          "Remove EOTA soft-threshold",     1),
    # Tier 2 — Semantic Region SWD ablations
    ("abl_k1_global",   "configs/semantic_swd_musiq/abl_k1_global.json",        "K=1 (global SWD only)",          2),
    ("abl_blend0",      "configs/semantic_swd_musiq/abl_blend0_pure_global.json","beta=0 pure global",            2),
    ("abl_blend1",      "configs/semantic_swd_musiq/abl_blend1_pure_region.json","beta=1 pure region",            2),
    ("abl_k64",         "configs/semantic_swd_musiq/abl_k64_extreme.json",      "K=64 extreme",                   2),
    # Tier 3 — architecture extremes
    ("abl_ll_w0",       "configs/semantic_swd_musiq/abl_ll_w0.json",            "lambda_LL=0 (no LL weight)",     3),
    ("abl_ll_w1",       "configs/semantic_swd_musiq/abl_ll_w1.json",            "lambda_LL=1.0 (full LL)",        3),
    ("abl_route_p05",   "configs/semantic_swd_musiq/abl_route_p05.json",        "route train_prob=0.5",           3),
    ("abl_route_p10",   "configs/semantic_swd_musiq/abl_route_p10.json",        "route train_prob=1.0",           3),
    # Tier 4 — mechanism replacements
    ("abl_soft_mask",   "configs/semantic_swd_musiq/abl_soft_mask.json",        "Soft mask instead of hard kmeans",4),
    ("abl_sinkhorn",    "configs/semantic_swd_musiq/abl_sinkhorn.json",         "Sinkhorn OT instead of SWD",     4),
    ("abl_spectral",    "configs/semantic_swd_musiq/abl_spectral.json",         "Spectral-decoupled region SWD",  4),
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


def run_one(name: str, config: str, desc: str, tier: int) -> bool:
    log(f"=== START [T{tier}] {name}: {desc} ===")
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
            log(f"=== DONE  [T{tier}] {name}: SUCCESS ({elapsed/60:.1f} min) ===")
            return True
        log(f"=== DONE  [T{tier}] {name}: FAILED rc={result.returncode} ({elapsed/60:.1f} min) ===")
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
        log(f"=== DONE  [T{tier}] {name}: TIMEOUT ({elapsed/60:.1f} min) ===")
        return False
    except Exception as e:
        elapsed = time.time() - start
        log(f"=== DONE  [T{tier}] {name}: ERROR {e} ({elapsed/60:.1f} min) ===")
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
        log(f"=== Queue start: {len(EXPERIMENTS)} experiments ===")
        log(f"CUDA_ALLOC_CONF: {os.environ.get('PYTORCH_CUDA_ALLOC_CONF', 'default')}")
        ok = 0
        fail = 0
        skip = 0
        for name, config, desc, tier in EXPERIMENTS:
            if is_complete(name):
                log(f"--- SKIP  [T{tier}] {name}: already complete ---")
                skip += 1
                continue
            if run_one(name, config, desc, tier):
                ok += 1
            else:
                fail += 1
            time.sleep(5)
        log(f"=== Queue done: {ok} ok, {fail} fail, {skip} skip ===")
        return 0 if fail == 0 else 2
    finally:
        if os.path.exists(PID):
            os.remove(PID)


if __name__ == "__main__":
    sys.exit(main())
