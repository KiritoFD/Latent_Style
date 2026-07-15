"""Monitor t11_repro eval, then auto-start t11e2 eval. Run on remote."""
import os
import sys
import json
import time
import subprocess

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, r"I:\Github\Latent_Style\SchrodingerBridge\src")

EXP_DIR = r"I:\Github\Latent_Style\SchrodingerBridge\exp"
LOG_DIR = r"C:\Users\Administrator\logs"

def check_summary(exp_name, ckpt_name):
    """Check if eval summary.json exists and return metrics."""
    path = os.path.join(EXP_DIR, exp_name, "full_eval", ckpt_name, "summary.json")
    if not os.path.isfile(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            s = json.load(f)
        ana = s.get("analysis", {})
        tr = ana.get("style_transfer_ability", {})
        ap = ana.get("all_pairs_overview", {})
        return {
            "transfer_clip": tr.get("clip_style"),
            "transfer_lpips": tr.get("content_lpips"),
            "allpairs_clip": ap.get("clip_style"),
            "allpairs_lpips": ap.get("content_lpips"),
            "wall_total": s.get("timings_sec", {}).get("wall_total"),
        }
    except Exception:
        return None

def wait_for_eval(exp_name, ckpt_name, log_name, timeout=1800):
    """Wait for eval to complete. Returns metrics or None on timeout."""
    print(f"Waiting for {exp_name}/{ckpt_name} eval to complete...")
    start = time.time()
    while time.time() - start < timeout:
        m = check_summary(exp_name, ckpt_name)
        if m:
            print(f"\n=== {exp_name}/{ckpt_name} EVAL COMPLETE ===")
            print(f"  transfer: clip={m['transfer_clip']:.4f}, lpips={m['transfer_lpips']:.4f}")
            print(f"  allpairs: clip={m['allpairs_clip']:.4f}, lpips={m['allpairs_lpips']:.4f}")
            print(f"  wall: {m['wall_total']:.1f}s")
            return m
        # Check progress from log
        log_path = os.path.join(LOG_DIR, f"{log_name}_fulleval.out")
        if os.path.isfile(log_path):
            with open(log_path, "rb") as f:
                data = f.read()
            text = data.decode("utf-8", errors="replace")
            lines = text.splitlines()
            batch_lines = [l for l in lines if "Generating Batch" in l]
            if batch_lines:
                last = batch_lines[-1]
                print(f"  [{int(time.time()-start)}s] {last.strip()[:80]}")
        time.sleep(30)
    print(f"TIMEOUT after {timeout}s")
    return None

def start_t11e2_eval():
    """Start t11e2 eval using wmic."""
    print("\n=== Starting t11e2 eval ===")
    bat_path = r"I:\Github\Latent_Style\SchrodingerBridge\scripts\_run_t11e2_eval.bat"
    r = subprocess.run(
        ["wmic", "process", "call", "create", f"'{bat_path}'"],
        capture_output=True, text=True, timeout=15
    )
    print(r.stdout)
    if r.returncode != 0:
        print(f"ERR: {r.stderr}")

if __name__ == "__main__":
    # 1. Wait for t11_repro eval
    print("=" * 60)
    print("MONITOR: t11_repro epoch_0005 eval")
    print("=" * 60)
    m1 = wait_for_eval("t11_repro_15ep", "epoch_0005", "t11_repro", timeout=1800)

    # 2. Start t11e2 eval
    start_t11e2_eval()
    time.sleep(10)  # Wait for process to start

    # 3. Wait for t11e2 eval
    print("\n" + "=" * 60)
    print("MONITOR: t11e2 epoch_0015 eval")
    print("=" * 60)
    m2 = wait_for_eval("t11e2_extrap05_15ep", "epoch_0015", "t11e2", timeout=1800)

    # 4. Summary comparison
    print("\n" + "=" * 60)
    print("=== FINAL COMPARISON ===")
    print("=" * 60)
    print(f"Original T11 (5ep, p=0.8):  clip=0.7213, lpips=0.2868")
    if m1:
        print(f"t11_repro (15ep, ep5):      clip={m1['allpairs_clip']:.4f}, lpips={m1['allpairs_lpips']:.4f}")
    else:
        print(f"t11_repro: FAILED/TIMEOUT")
    if m2:
        print(f"t11e2 (15ep, extrap=0.5):   clip={m2['allpairs_clip']:.4f}, lpips={m2['allpairs_lpips']:.4f}")
    else:
        print(f"t11e2: FAILED/TIMEOUT")
    print("=" * 60)
