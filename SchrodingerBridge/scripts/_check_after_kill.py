"""Check status after killing batch script. Run on remote."""
import os
import json
import sys

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

EXP_DIR = r"I:\Github\Latent_Style\SchrodingerBridge\exp"

# Check all t11 exp dirs
print("=" * 70)
print("=== T11 exp directories status ===")
print("=" * 70)
for d in sorted(os.listdir(EXP_DIR)):
    if "t11" in d.lower() and os.path.isdir(os.path.join(EXP_DIR, d)):
        full = os.path.join(EXP_DIR, d)
        contents = sorted(os.listdir(full))
        ckpts = [c for c in contents if c.endswith(".pt")]
        eval_dirs = [c for c in contents if "eval" in c.lower()]
        print(f"\n[{d}]")
        print(f"  checkpoints: {ckpts}")
        print(f"  eval dirs: {eval_dirs}")
        # Check for summary.json in eval dirs
        for ed in eval_dirs:
            ed_path = os.path.join(full, ed)
            for root, dirs, files in os.walk(ed_path):
                for f in files:
                    if f == "summary.json":
                        p = os.path.join(root, f)
                        try:
                            with open(p, "r", encoding="utf-8") as fh:
                                s = json.load(fh)
                            ana = s.get("analysis", {})
                            tr = ana.get("style_transfer_ability", {})
                            ap = ana.get("all_pairs_overview", {})
                            print(f"  {ed}/{os.path.basename(root)}: clip={tr.get('clip_style','?')}, lpips={tr.get('content_lpips','?')}")
                            print(f"    allpairs: clip={ap.get('clip_style','?')}, lpips={ap.get('content_lpips','?')}")
                        except Exception as e:
                            print(f"  {ed}/{os.path.basename(root)}: ERR {e}")

# Check t11e1 logs for failure reason
print("\n" + "=" * 70)
print("=== t11e1 failure check ===")
print("=" * 70)
t11e1_log = r"C:\Users\Administrator\logs\t11e1_ll05_15ep_train_eval.out"
if os.path.isfile(t11e1_log):
    with open(t11e1_log, "rb") as f:
        data = f.read()
    text = data.decode("utf-8", errors="replace")
    lines = text.splitlines()
    # Print last 30 lines
    print("--- last 30 lines ---")
    for line in lines[-30:]:
        print(line[:200])
    # Look for error
    for line in lines:
        if "error" in line.lower() or "traceback" in line.lower() or "fail" in line.lower() or "exception" in line.lower():
            print(f"\nERROR LINE: {line[:200]}")
else:
    print(f"[NOT FOUND: {t11e1_log}]")

# Check GPU memory
print("\n" + "=" * 70)
print("=== GPU memory ===")
print("=" * 70)
import subprocess
r = subprocess.run(["nvidia-smi", "--query-gpu=memory.used,memory.total", "--format=csv"],
                   capture_output=True, text=True, timeout=10)
print(r.stdout)
