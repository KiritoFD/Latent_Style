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
    """Parse ISO timestamp, return datetime or None."""
    for fmt in ["%Y-%m-%dT%H:%M:%S", "%Y-%m-%dT%H:%M:%S+08:00", "%Y-%m-%d %H:%M:%S"]:
        try:
            return datetime.strptime(s, fmt)
        except:
            pass
    try:
        return datetime.fromisoformat(s)
    except:
        return None

def probe_phase2(path):
    name = os.path.basename(path)
    if not os.path.isdir(path):
        print(f"{name}\t{path}\tNOT_FOUND\t\t\t\t\t\t\t\t")
        return

    mtime = run(f'stat -c "%y" "{path}"').split(".")[0]
    size_human = run(f'du -sh "{path}" 2>/dev/null', timeout=90)
    if size_human == "TIMEOUT" or not size_human:
        size_human = "?"

    train_sec = ""
    infer_sec = ""
    dataset = ""
    notes = ""

    # 1. Try to find corresponding _train.log in inmortal-exp/
    train_log = f"/mnt/i/Github/Latent_Style/exp_ours/recent/inmortal-exp/{name}_train.log"
    if os.path.exists(train_log):
        try:
            with open(train_log, 'r', errors='replace') as f:
                content = f.read()
            # Find ALL START/END pairs and sum durations
            start_pat = re.compile(r'===\s*START\s+(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\+08:00)', re.IGNORECASE)
            end_pat = re.compile(r'===\s*END\s+(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\+08:00)\s+rc=(\d+)', re.IGNORECASE)
            starts = [(m.start(), m.group(1)) for m in start_pat.finditer(content)]
            ends = [(m.start(), m.group(1), m.group(2)) for m in end_pat.finditer(content)]

            # Match starts with ends (first end after each start)
            total_dur = 0
            pair_count = 0
            first_start = None
            last_end = None
            for s_pos, s_ts in starts:
                # Find first end after this start
                for e_pos, e_ts, e_rc in ends:
                    if e_pos > s_pos:
                        t1 = parse_ts(s_ts)
                        t2 = parse_ts(e_ts)
                        if t1 and t2:
                            dur = (t2 - t1).total_seconds()
                            if dur > 0 and dur < 7200:  # skip absurdly long gaps (>2h = probably cross-day)
                                total_dur += dur
                                pair_count += 1
                                if first_start is None:
                                    first_start = t1
                                last_end = t2
                        break

            if total_dur > 0:
                train_sec = f"{total_dur:.0f}"
                notes = f"train.log({pair_count} runs)"
            elif first_start and last_end:
                # Fallback: first start to last end
                dur = (last_end - first_start).total_seconds()
                if dur > 0:
                    train_sec = f"{dur:.0f}"
                    notes = f"train.log(first-last)"
        except Exception as e:
            notes = f"train.log_err:{e}"

    # 2. Fallback: compute from eval summary.json timestamps
    if not train_sec:
        timestamps = []
        for root, dirs, files in os.walk(path):
            depth = root[len(path):].count(os.sep)
            if depth > 3:
                continue
            for f in files:
                if f == "summary.json":
                    full = os.path.join(root, f)
                    try:
                        with open(full) as fh:
                            data = json.load(fh)
                        if isinstance(data, dict) and "timestamp" in data:
                            ts = str(data["timestamp"])
                            t = parse_ts(ts)
                            if t:
                                timestamps.append(t)
                    except:
                        pass
        if len(timestamps) >= 2:
            timestamps.sort()
            dur = (timestamps[-1] - timestamps[0]).total_seconds()
            if dur > 0:
                train_sec = f"{dur:.0f}"
                notes = f"eval_ts({len(timestamps)} evals)"

    # 3. Also check for WALL_SECONDS in any .log file
    if not train_sec:
        all_logs = run(f'find "{path}" -maxdepth 2 -type f -name "*.log" 2>/dev/null', timeout=15)
        for f in all_logs.split("\n"):
            if not f or f == "TIMEOUT":
                continue
            try:
                sz = os.path.getsize(f)
                if sz > 5_000_000:
                    result = run(f'tail -5000 "{f}"', timeout=15)
                    lines = result.split("\n")
                else:
                    with open(f, 'r', errors='replace') as fh:
                        lines = fh.readlines()
                for line in reversed(lines):
                    m = re.search(r'WALL_SECONDS\s*=\s*([\d.]+)', line, re.IGNORECASE)
                    if m:
                        train_sec = m.group(1)
                        notes = f"log:{os.path.basename(f)}"
                        break
                if train_sec:
                    break
            except:
                pass

    # Count ckpts and images
    ckpt_count = run(f'find "{path}" -maxdepth 4 -type f \\( -name "*.ckpt" -o -name "*.pt" -o -name "*.pth" -o -name "*.safetensors" \\) 2>/dev/null | wc -l', timeout=30)
    image_count = run(f'find "{path}" -maxdepth 4 -type f \\( -name "*.png" -o -name "*.jpg" -o -name "*.jpeg" \\) 2>/dev/null | wc -l', timeout=30)
    if ckpt_count == "TIMEOUT":
        ckpt_count = "scan_timeout"
    if image_count == "TIMEOUT":
        image_count = "scan_timeout"

    print(f"{name}\t{path}\t{mtime}\t{size_human}\t{train_sec}\t{human(train_sec)}\t{infer_sec}\t{ckpt_count}\t{image_count}\t{dataset}\t{notes}")

for path in sys.argv[1:]:
    probe_phase2(path)
