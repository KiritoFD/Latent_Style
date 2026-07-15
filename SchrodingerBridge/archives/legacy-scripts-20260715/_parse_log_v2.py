"""Search log for tswd values - simple approach."""
import re

LOG = r"C:\Users\Administrator\logs\strong_ablation.out"

with open(LOG, "r", encoding="utf-8", errors="replace") as f:
    content = f.read()

# Find all tswd values
tswd_matches = re.findall(r"tswd=([\d.]+)", content)
print(f"Total tswd occurrences: {len(tswd_matches)}")
if tswd_matches:
    vals = [float(v) for v in tswd_matches]
    print(f"  first 10: {vals[:10]}")
    print(f"  last 10:  {vals[-10:]}")
    print(f"  max={max(vals):.4f}, min={min(vals):.4f}")
    nonzero = [v for v in vals if v > 0.001]
    print(f"  nonzero count: {len(nonzero)}")
    if nonzero:
        print(f"  nonzero first 5: {nonzero[:5]}")

# Find all lines containing "Epoch" and "tswd" to see training progress
lines = content.split("\n")
epoch_tswd = [l for l in lines if "Epoch" in l and "tswd=" in l]
print(f"\nEpoch+tswd lines: {len(epoch_tswd)}")
for l in epoch_tswd[:3]:
    print(f"  FIRST: {l.strip()[:200]}")
for l in epoch_tswd[-3:]:
    print(f"  LAST:  {l.strip()[:200]}")

# Also search for checkpoint saves to identify sections
ckpt_lines = [l for l in lines if "Saved checkpoint" in l and "abl_" in l]
print(f"\nCheckpoint saves:")
for l in ckpt_lines:
    print(f"  {l.strip()[:200]}")

# Check if "contract_family" or "spatial_bridge" appears in training start
contract_lines = [l for l in lines if "contract_family" in l.lower() or "spatial_bridge" in l.lower()]
print(f"\nContract family lines: {len(contract_lines)}")
for l in contract_lines[:5]:
    print(f"  {l.strip()[:200]}")
