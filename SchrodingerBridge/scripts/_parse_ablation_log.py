"""Parse strong_ablation.out - extract all tswd values with timestamps."""
import re

LOG = r"C:\Users\Administrator\logs\strong_ablation.out"

with open(LOG, "r", encoding="utf-8", errors="replace") as f:
    lines = f.readlines()

# Extract all lines with tswd= and checkpoint saves
print("=== All tswd values (first 10 and last 10 per checkpoint section) ===")
sections = []
current_section = "unknown"
section_tswds = {}

for line in lines:
    # Detect checkpoint saves to identify sections
    m = re.search(r"Saved checkpoint:.*?abl_(\w+)\\epoch_", line)
    if m:
        current_section = m.group(1)
        sections.append(current_section)
        section_tswds.setdefault(current_section, [])

    # Detect training start
    m2 = re.search(r"--config.*?abl_(\w+)\.json", line)
    if m2:
        current_section = m2.group(1)
        section_tswds.setdefault(current_section, [])

    # Extract tswd
    m3 = re.search(r"tswd=([\d.]+)", line)
    if m3:
        section_tswds.setdefault(current_section, []).append(float(m3.group(1)))

for name in ["swd_to_mse", "wo_wavelet", "wo_swd", "ll_equal"]:
    vals = section_tswds.get(name, [])
    if vals:
        print(f"\nabl_{name}: n={len(vals)}")
        print(f"  first 5: {vals[:5]}")
        print(f"  last 5:  {vals[-5:]}")
        print(f"  max={max(vals):.4f}, min={min(vals):.4f}")
    else:
        print(f"\nabl_{name}: NO TSWD VALUES")

# Also check: does the full model (t1_asg) have tswd values?
# Look for any tswd > 0
nonzero = [(i, v) for i, v in enumerate([(name, vals) for name, vals in section_tswds.items()]) if any(x > 0 for x in v[1])]
print(f"\nSections with nonzero tswd: {[v[0] for v in nonzero]}")
