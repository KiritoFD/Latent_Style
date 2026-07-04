import sys
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
path = sys.argv[1] if len(sys.argv) > 1 else "logs/latent256_train.log"
with open(path, "r", encoding="utf-8", errors="replace") as f:
    for i, line in enumerate(f):
        if "nan" in line.lower() and "loss=nan" in line:
            print(f"Line {i}: {line.rstrip()}")
            # Also print 3 lines before
            break
print("---")
# Find first non-nan after nan starts
with open(path, "r", encoding="utf-8", errors="replace") as f:
    lines = f.readlines()
    for i, line in enumerate(lines):
        if "loss=nan" in line and i > 0:
            print(f"BEFORE (line {i-1}): {lines[i-1].rstrip()}")
            print(f"NAN   (line {i}): {line.rstrip()}")
            break
