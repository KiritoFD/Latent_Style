import sys
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
path = r"I:\Github\Latent_Style\SchrodingerBridge\exp\710_b7_2res_canonical_dino_log.txt"
with open(path, "rb") as f:
    raw = f.read()
# Handle UTF-16 LE BOM
if raw[:2] == b"\xff\xfe":
    text = raw[2:].decode("utf-16-le", errors="replace")
else:
    text = raw.decode("utf-8", errors="replace")
lines = text.strip().split("\n")
for line in lines[-20:]:
    print(line.rstrip())
