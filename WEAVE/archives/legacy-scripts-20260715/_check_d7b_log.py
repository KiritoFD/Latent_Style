"""Check D7b training CSV log for NaN and summarize loss trend."""
import csv
import sys

path = r"I:\Github\Latent_Style\SchrodingerBridge\exp\d7b_deep_backbone_lr5e5\logs\training_20260710_023602.csv"

with open(path, "r", encoding="utf-8", errors="replace") as f:
    lines = f.readlines()

print(f"TOTAL_LINES={len(lines)}")
print(f"HEADER={lines[0].strip()[:200]}")
print(f"FIRST_DATA={lines[1].strip()[:300]}")
print(f"LAST_DATA={lines[-1].strip()[:300]}")

# Check for NaN
nan_count = 0
for i, line in enumerate(lines):
    if "nan" in line.lower():
        nan_count += 1
        if nan_count <= 3:
            print(f"NAN_AT_LINE_{i}: {line.strip()[:200]}")
print(f"TOTAL_NAN_LINES={nan_count}")

# Extract loss at each epoch boundary (epoch summary lines usually have epoch= in them)
print("\n=== EPOCH_SUMMARIES ===")
for i, line in enumerate(lines):
    if "epoch=" in line.lower() or (i > 0 and i % 52 == 0):
        # Try to extract loss and epoch info
        parts = line.strip().split(",")
        # Find loss and epoch fields
        loss_val = None
        epoch_val = None
        for p in parts:
            p = p.strip()
            if p.startswith("loss="):
                loss_val = p
            elif p.startswith("epoch=") or "epoch=" in p:
                epoch_val = p
        if loss_val or epoch_val:
            print(f"LINE_{i}: {loss_val} {epoch_val}")
