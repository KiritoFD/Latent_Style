"""Remote experiment runner - runs all mechanism + ablation experiments sequentially."""
import subprocess
import sys
import os
import time
import json
from datetime import datetime

REPO = r"I:\Github\Latent_Style\SchrodingerBridge"
PYTHON = r"C:\Program Files\Python312\python.exe"

# Infra: expandable_segments NOT supported on Windows CUDA allocator.
# Rely on conservative batch size instead. PYTORCH_CUDA_ALLOC_CONF left default.
# Empirical VRAM mapping on RTX 3060 12GB (bf16 AMP):
#   bs=120 -> 10.0GB (safe, baseline+simple ablations)
#   bs=128 -> 11.6GB (tight, succeeded for no_swd/no_dwt/no_wct/no_eota)
#   bs=136 -> 11.7GB (OOM for blend0/blend1 — SWD path memory varies)
# Choose bs=112 for headroom across ALL ablation configs (region_sinkhorn,
# region_spectral, K=64 all have different memory footprints).

# Experiments that already completed successfully at bs=128 (Tier 1 core removals).
# These are kept as-is: component-removal ablations where the relative gap is
# what matters, and bs=128 vs bs=112 introduces negligible variance for the
# large signal (-0.024 to -0.054 CLIP-S) these produce.
COMPLETED = {"abl_no_swd_loss", "abl_no_dwt_route", "abl_no_wct", "abl_no_eota"}

# Experiment queue: (name, config_path, description)
EXPERIMENTS = [
    # Baseline first (full WEAVE with bs=112)
    ("abl_baseline", "configs/semantic_swd_musiq/abl_baseline.json", "Full WEAVE baseline (bs=112)"),
    # Remaining ablations (Tier 2-4)
    ("abl_k1_global", "configs/semantic_swd_musiq/abl_k1_global.json", "K=1 global SWD"),
    ("abl_blend0", "configs/semantic_swd_musiq/abl_blend0_pure_global.json", "beta=0 pure global"),
    ("abl_blend1", "configs/semantic_swd_musiq/abl_blend1_pure_region.json", "beta=1 pure region"),
    ("abl_k64", "configs/semantic_swd_musiq/abl_k64_extreme.json", "K=64 extreme"),
    ("abl_soft_mask", "configs/semantic_swd_musiq/abl_soft_mask.json", "Soft mask"),
    ("abl_ll_w0", "configs/semantic_swd_musiq/abl_ll_w0.json", "lambda_LL=0"),
    ("abl_ll_w1", "configs/semantic_swd_musiq/abl_ll_w1.json", "lambda_LL=1.0"),
    ("abl_route_p05", "configs/semantic_swd_musiq/abl_route_p05.json", "route p=0.5"),
    ("abl_route_p10", "configs/semantic_swd_musiq/abl_route_p10.json", "route p=1.0"),
    ("abl_sinkhorn", "configs/semantic_swd_musiq/abl_sinkhorn.json", "Sinkhorn OT"),
    ("abl_spectral", "configs/semantic_swd_musiq/abl_spectral.json", "Spectral SWD (M5)"),
]

log_file = os.path.join(REPO, "remote_ablation_log.txt")
pid_file = os.path.join(REPO, "remote_ablation_runner.pid")

def log(msg):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    with open(log_file, 'a') as f:
        f.write(line + '\n')

def is_complete(name):
    """Check if experiment already has results."""
    if name in COMPLETED:
        return True
    # Check for summary.json in exp directory
    exp_dir = os.path.join(REPO, "exp", name, "full_eval")
    if not os.path.isdir(exp_dir):
        return False
    for epoch_dir in os.listdir(exp_dir):
        summary = os.path.join(exp_dir, epoch_dir, "summary.json")
        if os.path.exists(summary):
            try:
                with open(summary) as f:
                    data = json.load(f)
                # Verify it has actual metrics
                apo = data.get("analysis", {}).get("all_pairs_overview", {})
                if apo.get("clip_style") is not None:
                    return True
            except:
                pass
    return False

def run_experiment(name, config, desc):
    log(f"=== START {name}: {desc} ===")
    log(f"Config: {config}")
    start = time.time()
    exp_log = os.path.join(REPO, f"exp_log_{name}.txt")
    try:
        with open(exp_log, 'w') as stdout_f:
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
            log(f"=== DONE {name}: SUCCESS ({elapsed/60:.1f} min) ===")
        else:
            log(f"=== DONE {name}: FAILED (rc={result.returncode}, {elapsed/60:.1f} min) ===")
            stderr_tail = result.stderr[-2000:] if result.stderr else "(empty)"
            log(f"STDERR_TAIL: {stderr_tail}")
            stdout_tail = ""
            try:
                with open(exp_log, 'r') as f:
                    lines = f.readlines()
                    stdout_tail = ''.join(lines[-50:])
            except:
                pass
            if stdout_tail:
                log(f"STDOUT_TAIL: {stdout_tail}")
    except subprocess.TimeoutExpired:
        elapsed = time.time() - start
        log(f"=== DONE {name}: TIMEOUT ({elapsed/60:.1f} min) ===")
    except Exception as e:
        elapsed = time.time() - start
        log(f"=== DONE {name}: ERROR ({e}, {elapsed/60:.1f} min) ===")

if __name__ == "__main__":
    # Prevent duplicate runs
    if os.path.exists(pid_file):
        with open(pid_file, 'r') as f:
            old_pid = f.read().strip()
        log(f"FATAL: Another runner is already running (PID={old_pid}). Delete {pid_file} to force restart.")
        sys.exit(1)
    with open(pid_file, 'w') as f:
        f.write(str(os.getpid()))

    try:
        log(f"Starting remote experiment queue: {len(EXPERIMENTS)} experiments")
        log(f"CUDA_ALLOC_CONF: {os.environ.get('PYTORCH_CUDA_ALLOC_CONF', 'default')}")
        for name, config, desc in EXPERIMENTS:
            if is_complete(name):
                log(f"=== SKIP {name}: already complete ===")
                continue
            run_experiment(name, config, desc)
            time.sleep(5)
        log("=== ALL EXPERIMENTS COMPLETE ===")
    finally:
        if os.path.exists(pid_file):
            os.remove(pid_file)
