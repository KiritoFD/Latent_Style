import os, glob, sys, subprocess

EXP = "I:/GitHub/Latent_Style/SchrodingerBridge/exp"

def section(t):
    print("\n==== " + t + " ====")

# 1) identify pid 12880 (and any python)
section("Python processes")
try:
    import psutil
    for p in psutil.process_iter(["pid", "name", "cmdline", "create_time"]):
        if p.info["name"] and "python" in p.info["name"].lower():
            cl = " ".join(p.info["cmdline"] or [])
            print(f"  PID {p.info['pid']}: {cl[:160]}")
except Exception as e:
    print("  psutil unavailable:", e)
    # fallback: tasklist csv
    out = subprocess.run(["tasklist", "/v", "/fo", "csv"], capture_output=True, text=True).stdout
    for line in out.splitlines():
        if "python" in line.lower():
            print("  ", line[:200])

# 2) R5 test dir
section("R5 test dir")
r5 = "I:/datasets/wikiarts20_512_test"
if os.path.isdir(r5):
    sub = sorted([d for d in os.listdir(r5) if os.path.isdir(os.path.join(r5, d)) and not d.startswith(".")])
    print("  exists, subdirs:", sub)
    total_imgs = 0
    for s in sub:
        n = len([f for f in os.listdir(os.path.join(r5, s)) if f.lower().endswith((".png", ".jpg", ".jpeg"))])
        total_imgs += n
        print(f"    {s}: {n} images")
    print("  total images:", total_imgs)
else:
    print("  MISSING:", r5)

# 3) StyleAligned outputs
section("StyleAligned outputs")
for name in ["photo2art256", "random5"]:
    d = os.path.join(EXP, "baseline_stylealigned", name, "images")
    if os.path.isdir(d):
        n = len([f for f in os.listdir(d) if f.lower().endswith(".png")])
        print(f"  {name}: {n} png")
    else:
        print(f"  {name}: dir missing")

# 4) Z-STAR log + outputs
section("Z-STAR")
zl = os.path.join(EXP, "baseline_zstar", "zstar.log")
if os.path.isfile(zl):
    with open(zl, "r", encoding="utf-8", errors="ignore") as f:
        lines = f.read().splitlines()
    print("  zstar.log last 15 lines:")
    for l in lines[-15:]:
        print("    " + l[:160])
else:
    print("  no zstar.log")
for name in ["D5", "P2A", "R5"]:
    d = os.path.join(EXP, "baseline_zstar", name)
    if os.path.isdir(d):
        n = len([f for f in os.listdir(d) if f.lower().endswith(".png")])
        print(f"  out {name}: {n} png")
    else:
        print(f"  out {name}: missing")

# 5) flags
section("Flags")
for fl in [".zstar_launched", ".styleshot_launched"]:
    p = os.path.join(EXP, fl)
    print(f"  {fl}: {'PRESENT' if os.path.exists(p) else 'absent'}")

# 6) cleanup stale flag
section("Cleanup")
zp = os.path.join(EXP, ".zstar_launched")
if os.path.exists(zp):
    os.remove(zp)
    print("  removed stale .zstar_launched")
else:
    print("  .zstar_launched already absent")
sp = os.path.join(EXP, ".styleshot_launched")
if os.path.exists(sp):
    os.remove(sp)
    print("  removed stale .styleshot_launched")

print("\nDONE_DIAG")
