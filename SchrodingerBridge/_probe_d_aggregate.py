"""Aggregate training time across all sub-experiments in a meta directory."""
import os, sys, json, subprocess, re
from datetime import datetime

def run(cmd, timeout=25):
    try:
        r = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=timeout)
        return r.stdout.strip()
    except subprocess.TimeoutExpired:
        return "TIMEOUT"
    except Exception as e:
        return f"ERR:{e}"

def human(sec):
    if sec == "" or sec is None:
        return ""
    try:
        s = float(sec)
        if s < 60:
            return f"{s:.1f}s"
        elif s < 3600:
            return f"{s/60:.1f}min"
        else:
            return f"{s/3600:.2f}h"
    except:
        return ""

def parse_ts(s):
    for fmt in ["%Y-%m-%dT%H:%M:%S+08:00", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M:%S"]:
        try:
            return datetime.strptime(s, fmt)
        except:
            pass
    try:
        return datetime.fromisoformat(s)
    except:
        return None

def extract_train_seconds_from_log(filepath):
    """Extract training seconds from a single log file. Returns (seconds, source_note)."""
    try:
        sz = os.path.getsize(filepath)
        if sz > 5_000_000:
            result = run(f'tail -5000 "{filepath}"', timeout=15)
            if result == "TIMEOUT":
                return 0, ""
            lines = result.split("\n")
        else:
            with open(filepath, 'r', errors='replace') as f:
                lines = f.readlines()

        # Priority patterns (skip EVAL/INFER lines)
        TRAIN_HIGH = [
            re.compile(r'TRAIN_STEP_\d+_WALL_SECONDS\s*=\s*([\d.]+)', re.IGNORECASE),
            re.compile(r'TRAIN_WALL_SECONDS\s*=\s*([\d.]+)', re.IGNORECASE),
            re.compile(r'===\s*END\s+rc=\d+\s+elapsed_sec=([\d.]+)', re.IGNORECASE),
        ]
        TRAIN_MID = [
            re.compile(r'WALL_SECONDS\s*=\s*([\d.]+)', re.IGNORECASE),
        ]
        TRAIN_LOW = [
            re.compile(r'wall_seconds\s*[:=]\s*([\d.]+)', re.IGNORECASE),
            re.compile(r'train_seconds\s*[:=]\s*([\d.]+)', re.IGNORECASE),
            re.compile(r'elapsed_sec\s*[:=]\s*([\d.]+)', re.IGNORECASE),
        ]

        for patterns in [TRAIN_HIGH, TRAIN_MID, TRAIN_LOW]:
            for line in reversed(lines):
                ll = line.upper()
                if "EVAL" in ll or "INFER" in ll:
                    continue
                for p in patterns:
                    m = p.search(line)
                    if m:
                        try:
                            return float(m.group(1)), os.path.basename(filepath)
                        except:
                            pass
            # If high patterns found something, don't try mid/low
            # (we return inside the loop above on success)

        # Sum START/END pairs
        content = "".join(lines)
        start_pat = re.compile(r'===\s*START\s+(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\+08:00)', re.IGNORECASE)
        end_pat = re.compile(r'===\s*END\s+(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\+08:00)\s+rc=(\d+)', re.IGNORECASE)
        starts = [(m.start(), m.group(1)) for m in start_pat.finditer(content)]
        ends = [(m.start(), m.group(1), m.group(2)) for m in end_pat.finditer(content)]
        if starts and ends:
            total = 0
            for s_pos, s_ts in starts:
                for e_pos, e_ts, e_rc in ends:
                    if e_pos > s_pos:
                        t1 = parse_ts(s_ts)
                        t2 = parse_ts(e_ts)
                        if t1 and t2:
                            dur = (t2 - t1).total_seconds()
                            if 0 < dur < 7200:
                                total += dur
                        break
            if total > 0:
                return total, f"START-END:{os.path.basename(filepath)}"

        # Final fallback: first/last Python logging timestamp
        # Pattern: "2026-06-21 07:10:37,295 - INFO - ..."
        ts_line_pat = re.compile(r'^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})', re.MULTILINE)
        all_ts = ts_line_pat.findall(content)
        if len(all_ts) >= 2:
            t1 = parse_ts(all_ts[0])
            t2 = parse_ts(all_ts[-1])
            if t1 and t2:
                dur = (t2 - t1).total_seconds()
                if 0 < dur < 86400:  # < 24h sanity check
                    return dur, f"first-last ts:{os.path.basename(filepath)}"

        return 0, ""
    except Exception:
        return 0, ""

def aggregate_dir(path):
    """Walk through a directory and sum training times from all logs."""
    name = os.path.basename(path)
    if not os.path.isdir(path):
        print(f"{name}\t{path}\tNOT_FOUND\t\t\t\t\t\t\t\t")
        return

    mtime = run(f'stat -c "%y" "{path}"').split(".")[0]
    size_human = run(f'du -sh "{path}" 2>/dev/null', timeout=120)
    if size_human == "TIMEOUT" or not size_human:
        size_human = "?"

    # Find all train.log files at depth 1 (inmortal-exp) or depth 2 (620_spatial_bridge)
    train_logs_result = run(f'find "{path}" -maxdepth 3 -type f -name "*train*.log" 2>/dev/null', timeout=30)
    train_logs = [f for f in train_logs_result.split("\n") if f and f != "TIMEOUT"] if train_logs_result else []

    total_sec = 0.0
    matched = 0
    samples = []
    for log in train_logs:
        sec, src = extract_train_seconds_from_log(log)
        if sec > 0:
            total_sec += sec
            matched += 1
            if len(samples) < 3:
                samples.append(os.path.basename(os.path.dirname(log) if "_train" in os.path.basename(log) else log))

    # Count ckpts and images
    ckpt_count = run(f'find "{path}" -maxdepth 4 -type f \\( -name "*.ckpt" -o -name "*.pt" -o -name "*.pth" -o -name "*.safetensors" \\) 2>/dev/null | wc -l', timeout=30)
    image_count = run(f'find "{path}" -maxdepth 4 -type f \\( -name "*.png" -o -name "*.jpg" -o -name "*.jpeg" \\) 2>/dev/null | wc -l', timeout=30)
    if ckpt_count == "TIMEOUT": ckpt_count = "scan_timeout"
    if image_count == "TIMEOUT": image_count = "scan_timeout"

    notes = f"aggregate({matched}/{len(train_logs)} logs)"
    train_sec_str = f"{total_sec:.0f}" if total_sec > 0 else ""

    print(f"{name}\t{path}\t{mtime}\t{size_human}\t{train_sec_str}\t{human(train_sec_str)}\t\t{ckpt_count}\t{image_count}\t\t{notes}")

for path in sys.argv[1:]:
    aggregate_dir(path)
