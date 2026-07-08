import os, subprocess, time
from pathlib import Path

EXP = Path("I:/GitHub/Latent_Style/SchrodingerBridge/exp")

def count(sub):
    d = EXP / sub / "images"
    return len([f for f in os.listdir(d) if f.endswith(".png")]) if d.is_dir() else 0

def task_mode(name):
    out = subprocess.run(["schtasks", "/Query", "/TN", name, "/FO", "LIST"],
                         capture_output=True, text=True).stdout
    for line in out.splitlines():
        if line.startswith("模式") or line.lstrip().startswith("模式"):
            return line.split(":", 1)[-1].strip()
    return "?"

print("=== StyleAligned ===")
for sub in ["photo2art256", "random5"]:
    print("  ", sub, count("baseline_stylealigned/" + sub))
md = EXP / "baseline_stylealigned" / "metadata.json"
print("  metadata.json exists:", md.exists())

print("=== Z-STAR ===")
for sub in ["D5", "P2A", "R5"]:
    print("  ", sub, count("baseline_zstar/" + sub))

print("=== StyleShot ===")
for sub in ["D5", "P2A", "R5"]:
    print("  ", sub, count("baseline_styleshot/" + sub))

print("=== Task modes ===")
for t in ["StyleAligned_Runs", "ZSTAR_Runs", "StyleShot_Runs", "Watchdog_Baselines"]:
    print("  ", t, task_mode(t))

print("=== Flags ===")
for f in [".zstar_launched", ".styleshot_launched"]:
    print("  ", f, (EXP / f).exists())

print("=== Python procs ===")
out = subprocess.run(["tasklist"], capture_output=True, text=True).stdout
for line in out.splitlines():
    if "python" in line.lower():
        print("  ", line.split()[0], line.split()[-2] if len(line.split()) > 2 else "")
print("done", time.strftime("%H:%M:%S"))
