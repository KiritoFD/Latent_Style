import sys
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
path = sys.argv[1]
with open(path, "rb") as f:
    raw = f.read()
if raw[:2] == b"\xff\xfe":
    text = raw[2:].decode("utf-16-le", errors="replace")
else:
    text = raw.decode("utf-8", errors="replace")
lines = text.strip().split("\n")
for line in lines[-15:]:
    print(line.rstrip())
