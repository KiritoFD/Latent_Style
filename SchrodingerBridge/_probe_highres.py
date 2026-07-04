"""Probe highres logs for training time."""
import os, re, sys
from datetime import datetime

def parse_ts_fmt(s, fmt):
    try:
        return datetime.strptime(s, fmt)
    except:
        return None

def parse_ts_any(s):
    """Try multiple timestamp formats."""
    for fmt in [
        "%Y-%m-%d %H:%M:%S",
        "%Y/%m/%d %H:%M:%S",
        "%Y/%m/%d %H:%M:%S.%f",
        "%Y-%m-%d %H:%M:%S.%f",
        "%Y-%m-%dT%H:%M:%S",
        "%Y-%m-%dT%H:%M:%S+08:00",
    ]:
        t = parse_ts_fmt(s, fmt)
        if t:
            return t
    try:
        return datetime.fromisoformat(s)
    except:
        return None

# Patterns for various timestamp formats
TS_PATTERNS = [
    # [2026-05-20 05:36:44] text
    re.compile(r'^\[(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})\]'),
    # [2026/05/18 周一 19:58:04.40] text  (Chinese weekday)
    re.compile(r'^\[(\d{4}/\d{2}/\d{2}) [^\]]+? (\d{2}:\d{2}:\d{2}\.\d+)\]'),
    # 2026-06-21 07:10:37,295 - INFO - text
    re.compile(r'^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})'),
    # 2026/05/18 周一 19:02:18.47 text
    re.compile(r'^(\d{4}/\d{2}/\d{2} [^\s]+ \d{2}:\d{2}:\d{2}\.\d+)'),
    # 2026-05-20T05:36:44
    re.compile(r'^(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2})'),
]

def extract_first_last_ts(filepath):
    """Extract first and last timestamps from a log file."""
    try:
        with open(filepath, 'r', errors='replace') as f:
            lines = f.readlines()
        first_ts = None
        last_ts = None
        for line in lines:
            for pat in TS_PATTERNS:
                m = pat.match(line)
                if m:
                    raw = m.group(1)
                    # Handle Chinese weekday format: "2026/05/18 周一 19:58:04.40"
                    if '/' in raw and ' ' not in raw:
                        # Need second group from pattern 2
                        m2 = re.match(r'^(\d{4}/\d{2}/\d{2}) [^\]]+? (\d{2}:\d{2}:\d{2}\.\d+)$', raw + ' ' + line[m.end(1):m.end(1)+15] if len(line) > m.end(1)+15 else raw)
                        # Just reconstruct
                        pass
                    t = parse_ts_any(raw)
                    if t is None:
                        # Try combining date+time for Chinese weekday format
                        if '/' in raw:
                            # Try just date+time portion
                            parts = raw.split()
                            if len(parts) >= 1:
                                t = parse_ts_fmt(parts[0], "%Y/%m/%d")
                    if t:
                        if first_ts is None:
                            first_ts = t
                        last_ts = t
                    break
        if first_ts and last_ts:
            return first_ts, last_ts
        return None, None
    except Exception as e:
        return None, None

def probe_log(filepath):
    """Probe a single log file."""
    name = os.path.basename(filepath)
    sz = os.path.getsize(filepath)
    print(f"\n=== {name} ({sz} bytes) ===")
    first, last = extract_first_last_ts(filepath)
    if first and last:
        dur = (last - first).total_seconds()
        print(f"  first_ts: {first}")
        print(f"  last_ts:  {last}")
        print(f"  duration: {dur:.0f}s ({dur/3600:.2f}h)")
    else:
        print(f"  no timestamps found")

# Also try matching Chinese weekday pattern manually
def extract_chinese_weekday_ts(filepath):
    """Extract timestamps from Chinese weekday format logs."""
    pat = re.compile(r'(\d{4})/(\d{2})/(\d{2}) [^\s]+ (\d{2}):(\d{2}):(\d{2})\.(\d+)')
    try:
        with open(filepath, 'r', errors='replace') as f:
            content = f.read()
        matches = pat.findall(content)
        if not matches:
            return None, None
        timestamps = []
        for y, mo, d, h, mi, s, ms in matches:
            try:
                t = datetime(int(y), int(mo), int(d), int(h), int(mi), int(s))
                timestamps.append(t)
            except:
                pass
        if len(timestamps) >= 2:
            return timestamps[0], timestamps[-1]
        return None, None
    except:
        return None, None

logs = [
    "/mnt/i/Github/Latent_Style/exp_ours/recent/highres/s2wat_pipeline.log",
    "/mnt/i/Github/Latent_Style/exp_ours/recent/highres/s2wat_training.log",
    "/mnt/i/Github/Latent_Style/exp_ours/recent/highres/samst_train.log",
    "/mnt/i/Github/Latent_Style/exp_ours/recent/highres/samst_training.log",
    "/mnt/i/Github/Latent_Style/exp_ours/recent/highres/samst_all.log",
    "/mnt/i/Github/Latent_Style/exp_ours/recent/highres/samst_pipeline.log",
    "/mnt/i/Github/Latent_Style/exp_ours/recent/highres/samst_finish_then_s2wat.log",
]

total_sec = 0
print("=== Highres log analysis ===")
for log in logs:
    if not os.path.exists(log):
        continue
    # Try standard method first
    first, last = extract_first_last_ts(log)
    src = "ts_pattern"
    # Try Chinese weekday as fallback
    if not first:
        first, last = extract_chinese_weekday_ts(log)
        src = "cn_weekday"
    if first and last:
        dur = (last - first).total_seconds()
        if 0 < dur < 86400:
            total_sec += dur
            print(f"{os.path.basename(log)}: {first} -> {last} = {dur:.0f}s ({dur/3600:.2f}h) [{src}]")
        else:
            print(f"{os.path.basename(log)}: {first} -> {last} = {dur:.0f}s (skipped, out of range) [{src}]")
    else:
        print(f"{os.path.basename(log)}: no timestamps found")

print(f"\nTotal aggregate training time (highres): {total_sec:.0f}s ({total_sec/3600:.2f}h)")
