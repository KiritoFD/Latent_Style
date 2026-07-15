"""Remote check for T11 evo status. Run on remote Windows."""
import os
import json
import sys
import glob

# Force UTF-8 output on Windows
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

LOG_DIR = r"C:\Users\Administrator\logs"
EXP_DIR = r"I:\Github\Latent_Style\SchrodingerBridge\exp"

def tail(path, n=50):
    try:
        with open(path, "rb") as f:
            data = f.read()
        try:
            text = data.decode("utf-8", errors="replace")
        except Exception:
            text = data.decode("gbk", errors="replace")
        lines = text.splitlines()
        return "\n".join(lines[-n:])
    except Exception as e:
        return f"[ERR reading {path}: {e}]"

print("=" * 70)
print("=== t11_repro_15ep_train_eval.out (tail 60) ===")
print("=" * 70)
print(tail(os.path.join(LOG_DIR, "t11_repro_15ep_train_eval.out"), 60))

print("\n" + "=" * 70)
print("=== t11e1_ll05_15ep_train_eval.out (tail 40) ===")
print("=" * 70)
print(tail(os.path.join(LOG_DIR, "t11e1_ll05_15ep_train_eval.out"), 40))

print("\n" + "=" * 70)
print("=== t11e2_extrap05_15ep_train_eval.out (tail 40) ===")
print("=" * 70)
print(tail(os.path.join(LOG_DIR, "t11e2_extrap05_15ep_train_eval.out"), 40))

print("\n" + "=" * 70)
print("=== t11e3_p09_15ep_train_eval.out (tail 40) ===")
print("=" * 70)
print(tail(os.path.join(LOG_DIR, "t11e3_p09_15ep_train_eval.out"), 40))

# List exp dirs with t11
print("\n" + "=" * 70)
print("=== T11 exp directories ===")
print("=" * 70)
if os.path.isdir(EXP_DIR):
    for d in sorted(os.listdir(EXP_DIR)):
        if "t11" in d.lower() and os.path.isdir(os.path.join(EXP_DIR, d)):
            full = os.path.join(EXP_DIR, d)
            contents = sorted(os.listdir(full))
            print(f"\n[{d}] ({len(contents)} items)")
            for c in contents[:10]:
                print(f"  {c}")

# Find summary.json
print("\n" + "=" * 70)
print("=== summary.json files in T11 exp dirs ===")
print("=" * 70)
if os.path.isdir(EXP_DIR):
    for d in sorted(os.listdir(EXP_DIR)):
        if "t11" in d.lower() and os.path.isdir(os.path.join(EXP_DIR, d)):
            for root, dirs, files in os.walk(os.path.join(EXP_DIR, d)):
                for f in files:
                    if f == "summary.json":
                        p = os.path.join(root, f)
                        try:
                            with open(p, "r", encoding="utf-8") as fh:
                                s = json.load(fh)
                            ana = s.get("analysis", {})
                            tr = ana.get("style_transfer_ability", {})
                            ap = ana.get("all_pairs_overview", {})
                            print(f"\n[{d}] {p}")
                            print(f"  transfer: clip_style={tr.get('clip_style','?')}, content_lpips={tr.get('content_lpips','?')}")
                            print(f"  allpairs: clip_style={ap.get('clip_style','?')}, content_lpips={ap.get('content_lpips','?')}")
                        except Exception as e:
                            print(f"[ERR {p}: {e}]")

# Check running python processes
print("\n" + "=" * 70)
print("=== Running python processes ===")
print("=" * 70)
import subprocess
r = subprocess.run(["tasklist", "/FI", "IMAGENAME eq python.exe"], capture_output=True, text=True, timeout=10)
print(r.stdout)
