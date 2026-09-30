"""Compute B0 training time from log timestamps (UTF-16 encoded)."""
import re
from datetime import datetime

log = r"I:\Github\Latent_Style\SchrodingerBridge\exp\710_b0_t11_log.txt"
with open(log, "rb") as f:
    raw = f.read()
# Detect encoding via BOM
if raw[:2] == b"\xff\xfe":
    text = raw.decode("utf-16-le", errors="ignore")
elif raw[:2] == b"\xfe\xff":
    text = raw.decode("utf-16-be", errors="ignore")
else:
    text = raw.decode("utf-8", errors="ignore")

# Find first and last timestamps
ts_pattern = re.compile(r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})")
matches = ts_pattern.findall(text)
print(f"Found {len(matches)} timestamps")
if matches:
    first_ts = matches[0]
    last_ts = matches[-1]
    fmt = "%Y-%m-%d %H:%M:%S"
    t0 = datetime.strptime(first_ts, fmt)
    t1 = datetime.strptime(last_ts, fmt)
    delta = (t1 - t0).total_seconds() / 60.0
    print(f"B0 first_ts={first_ts} last_ts={last_ts}")
    print(f"B0 train_min={delta:.1f}")
else:
    print("No timestamps found")
