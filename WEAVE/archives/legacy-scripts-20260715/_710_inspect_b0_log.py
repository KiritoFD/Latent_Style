"""Inspect B0 log head and search for timestamps."""
import re

log = r"I:\Github\Latent_Style\SchrodingerBridge\exp\710_b0_t11_log.txt"
with open(log, "rb") as f:
    raw = f.read(2000)
print("=== First 500 bytes (repr) ===")
print(repr(raw[:500]))

# Try utf-8 decode
text = raw.decode("utf-8", errors="ignore")
print("\n=== First 500 chars ===")
print(text[:500])

# Search for any timestamp-like pattern
with open(log, "rb") as f:
    full = f.read().decode("utf-8", errors="ignore")
print(f"\n=== Total length: {len(full)} chars ===")
# Find all date-like patterns
patterns = [
    r"\d{4}-\d{2}-\d{2}",
    r"\d{2}:\d{2}:\d{2}",
    r"\d{4}/\d{2}/\d{2}",
]
for p in patterns:
    matches = re.findall(p, full)
    print(f"Pattern {p}: {len(matches)} matches, first 3: {matches[:3]}, last 3: {matches[-3:]}")

# Find lines with "INFO" or "Training"
for keyword in ["Training", "INFO", "Epoch 1", "Epoch 5"]:
    idx = full.find(keyword)
    if idx >= 0:
        print(f"\n=== First '{keyword}' at pos {idx} ===")
        print(repr(full[max(0,idx-50):idx+100]))
