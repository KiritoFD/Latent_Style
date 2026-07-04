import sys
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
path = sys.argv[1] if len(sys.argv) > 1 else "logs/latent256_train.log"
n = int(sys.argv[2]) if len(sys.argv) > 2 else 60
offset = int(sys.argv[3]) if len(sys.argv) > 3 else 0
with open(path, "r", encoding="utf-8", errors="replace") as f:
    for i, line in enumerate(f):
        if i < offset:
            continue
        if i >= offset + n:
            break
        print(line.rstrip())
