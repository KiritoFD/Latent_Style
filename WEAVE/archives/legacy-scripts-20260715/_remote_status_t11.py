"""Check t11_repro and t11e2 full_eval status on remote."""
import sys, os, json
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

LOG_DIR = r"C:\Users\Administrator\logs"
EXP_DIR = r"I:\Github\Latent_Style\SchrodingerBridge\exp"

print("=== t11_repro_fulleval.out (tail 40) ===")
try:
    with open(os.path.join(LOG_DIR, "t11_repro_fulleval.out"), "r", encoding="utf-8", errors="replace") as f:
        lines = f.readlines()
    for l in lines[-40:]:
        print(l.rstrip())
except Exception as e:
    print(f"ERR: {e}")

print("\n=== t11_repro_fulleval.err (tail 20) ===")
try:
    with open(os.path.join(LOG_DIR, "t11_repro_fulleval.err"), "r", encoding="utf-8", errors="replace") as f:
        lines = f.readlines()
    for l in lines[-20:]:
        print(l.rstrip())
except Exception as e:
    print(f"ERR: {e}")

print("\n=== t11e2_fulleval.out (tail 40) ===")
try:
    with open(os.path.join(LOG_DIR, "t11e2_fulleval.out"), "r", encoding="utf-8", errors="replace") as f:
        lines = f.readlines()
    for l in lines[-40:]:
        print(l.rstrip())
except Exception as e:
    print(f"ERR: {e}")

print("\n=== t11_repro summary.json ===")
for ep in ["epoch_0005", "epoch_0015"]:
    p = os.path.join(EXP_DIR, "t11_repro_15ep", "full_eval", ep, "summary.json")
    if os.path.exists(p):
        try:
            with open(p, "r", encoding="utf-8") as f:
                d = json.load(f)
            print(f"[{ep}] {json.dumps(d, ensure_ascii=False)}")
        except Exception as e:
            print(f"[{ep}] ERR: {e}")
    else:
        print(f"[{ep}] NOT FOUND: {p}")

print("\n=== t11e2 summary.json ===")
for ep in ["epoch_0005", "epoch_0010", "epoch_0015"]:
    p = os.path.join(EXP_DIR, "t11e2_extrap05_15ep", "full_eval", ep, "summary.json")
    if os.path.exists(p):
        try:
            with open(p, "r", encoding="utf-8") as f:
                d = json.load(f)
            print(f"[{ep}] {json.dumps(d, ensure_ascii=False)}")
        except Exception as e:
            print(f"[{ep}] ERR: {e}")
    else:
        print(f"[{ep}] NOT FOUND: {p}")

print("\n=== monitor_and_chain_evals.out (tail 20) ===")
try:
    with open(os.path.join(LOG_DIR, "monitor_and_chain_evals.out"), "r", encoding="utf-8", errors="replace") as f:
        lines = f.readlines()
    for l in lines[-20:]:
        print(l.rstrip())
except Exception as e:
    print(f"ERR: {e}")

print("\n=== python processes ===")
import subprocess
r = subprocess.run(["wmic", "process", "where", "name='python.exe'", "get", "ProcessId,CommandLine"],
                   capture_output=True, text=True, errors="replace")
print(r.stdout[:2000] if r.stdout else "(empty)")
