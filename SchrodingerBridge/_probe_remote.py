import os, sys, json, subprocess, re

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

def extract_from_json(data):
    train_sec = ""; infer_sec = ""; dataset = ""; extra = {}
    if not isinstance(data, dict):
        return train_sec, infer_sec, dataset, extra
    for k, v in data.items():
        kl = k.lower()
        if not isinstance(v, (int, float, str)):
            continue
        if "wall_second" in kl or ("wall" in kl and "second" in kl) or "wall_time" in kl or "walltime" in kl:
            train_sec = v
        elif "train_second" in kl or "training_second" in kl or "train_time" in kl:
            train_sec = v
        elif "total_second" in kl or "total_time" in kl:
            if train_sec == "": train_sec = v
        elif kl in ("wall", "elapsed", "elapsed_sec", "elapsed_seconds"):
            train_sec = v
        if "infer" in kl and ("second" in kl or "time" in kl):
            infer_sec = v
        elif "eval_second" in kl or "eval_time" in kl:
            infer_sec = v
        if kl in ("dataset", "dataset_name", "data_name", "dataset_dir"):
            dataset = str(v)
        if kl in ("step", "steps", "global_step", "epoch", "epochs", "total_steps"):
            extra[kl] = v
    return train_sec, infer_sec, dataset, extra

TIME_PATTERNS = [
    re.compile(r'TRAIN_STEP_\d+_WALL_SECONDS\s*=\s*([\d.]+)', re.IGNORECASE),
    re.compile(r'TRAIN_WALL_SECONDS\s*=\s*([\d.]+)', re.IGNORECASE),
    re.compile(r'WALL_SECONDS\s*=\s*([\d.]+)', re.IGNORECASE),
    re.compile(r'wall_seconds\s*[:=]\s*([\d.]+)', re.IGNORECASE),
    re.compile(r'train_seconds\s*[:=]\s*([\d.]+)', re.IGNORECASE),
    re.compile(r'training_seconds\s*[:=]\s*([\d.]+)', re.IGNORECASE),
    re.compile(r'total_seconds\s*[:=]\s*([\d.]+)', re.IGNORECASE),
    re.compile(r'train_time\s*[:=]\s*([\d.]+)', re.IGNORECASE),
    re.compile(r'elapsed_sec\s*[:=]\s*([\d.]+)', re.IGNORECASE),
    re.compile(r'elapsed_seconds\s*[:=]\s*([\d.]+)', re.IGNORECASE),
    re.compile(r'elapsed\s*[:=]\s*([\d.]+)\s*s', re.IGNORECASE),
    re.compile(r'"wall_seconds"\s*:\s*([\d.]+)', re.IGNORECASE),
    re.compile(r'"train_seconds"\s*:\s*([\d.]+)', re.IGNORECASE),
    re.compile(r'"total_seconds"\s*:\s*([\d.]+)', re.IGNORECASE),
    re.compile(r'"elapsed"\s*:\s*([\d.]+)', re.IGNORECASE),
    re.compile(r'"wall_time"\s*:\s*([\d.]+)', re.IGNORECASE),
    re.compile(r'"elapsed_sec"\s*:\s*([\d.]+)', re.IGNORECASE),
]
INFER_PATTERNS = [
    re.compile(r'EVAL_STEP_\d+_WALL_SECONDS\s*=\s*([\d.]+)', re.IGNORECASE),
    re.compile(r'EVAL_WALL_SECONDS\s*=\s*([\d.]+)', re.IGNORECASE),
    re.compile(r'inference_seconds\s*[:=]\s*([\d.]+)', re.IGNORECASE),
    re.compile(r'infer_seconds\s*[:=]\s*([\d.]+)', re.IGNORECASE),
    re.compile(r'eval_seconds\s*[:=]\s*([\d.]+)', re.IGNORECASE),
    re.compile(r'inference_time\s*[:=]\s*([\d.]+)', re.IGNORECASE),
]

def search_log_for_time(filepath):
    train_sec = ""; infer_sec = ""; notes = ""
    try:
        sz = os.path.getsize(filepath)
        if sz > 5_000_000:
            result = run(f'tail -5000 "{filepath}"', timeout=15)
            if result == "TIMEOUT":
                return "", "", "log_timeout"
            lines = result.split("\n")
        else:
            with open(filepath, 'r', errors='replace') as f:
                lines = f.readlines()

        # Priority-ordered train time patterns (search from end)
        TRAIN_HIGH_PATTERNS = [
            re.compile(r'TRAIN_STEP_\d+_WALL_SECONDS\s*=\s*([\d.]+)', re.IGNORECASE),
            re.compile(r'TRAIN_WALL_SECONDS\s*=\s*([\d.]+)', re.IGNORECASE),
            re.compile(r'===\s*END\s+rc=\d+\s+elapsed_sec=([\d.]+)', re.IGNORECASE),
        ]
        TRAIN_MID_PATTERNS = [
            re.compile(r'WALL_SECONDS\s*=\s*([\d.]+)', re.IGNORECASE),
        ]
        TRAIN_LOW_PATTERNS = [
            re.compile(r'wall_seconds\s*[:=]\s*([\d.]+)', re.IGNORECASE),
            re.compile(r'train_seconds\s*[:=]\s*([\d.]+)', re.IGNORECASE),
            re.compile(r'training_seconds\s*[:=]\s*([\d.]+)', re.IGNORECASE),
            re.compile(r'total_seconds\s*[:=]\s*([\d.]+)', re.IGNORECASE),
            re.compile(r'train_time\s*[:=]\s*([\d.]+)', re.IGNORECASE),
            re.compile(r'elapsed_sec\s*[:=]\s*([\d.]+)', re.IGNORECASE),
        ]

        # Pass 1: high-priority train patterns (skip EVAL/INFER lines)
        for line in reversed(lines):
            ll = line.upper()
            if "EVAL" in ll or "INFER" in ll:
                continue
            for p in TRAIN_HIGH_PATTERNS:
                m = p.search(line)
                if m:
                    train_sec = m.group(1)
                    notes = f"log:{os.path.basename(filepath)}"
                    break
            if train_sec:
                break

        # Pass 2: mid-priority (WALL_SECONDS without EVAL prefix)
        if not train_sec:
            for line in reversed(lines):
                ll = line.upper()
                if "EVAL" in ll or "INFER" in ll:
                    continue
                for p in TRAIN_MID_PATTERNS:
                    m = p.search(line)
                    if m:
                        train_sec = m.group(1)
                        notes = f"log:{os.path.basename(filepath)}"
                        break
                if train_sec:
                    break

        # Pass 3: low-priority (elapsed_sec etc.) - only if no WALL_SECONDS found
        if not train_sec:
            for line in reversed(lines):
                ll = line.upper()
                if "EVAL" in ll or "INFER" in ll:
                    continue
                for p in TRAIN_LOW_PATTERNS:
                    m = p.search(line)
                    if m:
                        train_sec = m.group(1)
                        notes = f"log:{os.path.basename(filepath)}"
                        break
                if train_sec:
                    break

        # Search for inference time (from end)
        for line in reversed(lines):
            for p in INFER_PATTERNS:
                m = p.search(line)
                if m:
                    infer_sec = m.group(1)
                    break
            if infer_sec:
                break

        return train_sec, infer_sec, notes
    except Exception as e:
        return "", "", f"log_err:{e}"

def probe(path):
    name = os.path.basename(path)
    if not os.path.isdir(path):
        print(f"{name}\t{path}\tNOT_FOUND\t\t\t\t\t\t\t\t")
        return
    mtime = run(f'stat -c "%y" "{path}"').split(".")[0]
    size_human = run(f'du -sh "{path}" 2>/dev/null', timeout=90)
    if size_human == "TIMEOUT" or not size_human:
        size_human = "?"

    find_result = run(f'find "{path}" -maxdepth 2 -type f \\( -name "summary.json" -o -name "train_log.json" -o -name "progress.json" -o -name "progress.log" -o -name "train.log" -o -name "keepalive.log" -o -name "metrics.csv" -o -name "run_meta.json" -o -name "remote_launcher.log" -o -name "launcher.log" \\) 2>/dev/null', timeout=20)
    meta_files = [f for f in find_result.split("\n") if f and f != "TIMEOUT"] if find_result else []

    train_sec = ""; infer_sec = ""; dataset = ""; notes = ""

    # First: extract dataset from run_meta.json if present
    for f in meta_files:
        if os.path.basename(f) == "run_meta.json":
            try:
                with open(f) as fh:
                    data = json.load(fh)
                if isinstance(data, dict) and "dataset" in data:
                    dataset = str(data["dataset"])
            except:
                pass

    priority = []
    for f in meta_files:
        bn = os.path.basename(f)
        if bn == "summary.json": priority.append((0, f))
        elif bn == "progress.json": priority.append((1, f))
        elif bn == "train_log.json": priority.append((2, f))
    priority.sort()
    for prio, f in priority:
        try:
            with open(f) as fh:
                content = fh.read()
            data = None
            try:
                data = json.loads(content)
            except:
                lns = [l for l in content.strip().split("\n") if l.strip()]
                if lns:
                    try: data = json.loads(lns[-1])
                    except: pass
            if data is None: continue
            if isinstance(data, list):
                data = data[-1] if data else {}
            ts, isec, ds, extra = extract_from_json(data)
            if ts: train_sec = ts
            if isec: infer_sec = isec
            if ds and not dataset: dataset = ds
            notes = f"src={os.path.basename(f)}"
            if extra:
                notes += " " + " ".join(f"{k}={v}" for k,v in extra.items())
            break
        except Exception as e:
            notes = f"read_err:{e}"

    if not train_sec:
        log_priority = []
        for f in meta_files:
            bn = os.path.basename(f)
            if bn == "progress.log": log_priority.append((0, f))
            elif bn == "train.log": log_priority.append((1, f))
            elif bn == "keepalive.log": log_priority.append((2, f))
            elif bn == "remote_launcher.log": log_priority.append((3, f))
            elif bn == "launcher.log": log_priority.append((4, f))
            elif bn == "segmented.log": log_priority.append((5, f))
        log_priority.sort()
        for prio, f in log_priority:
            ts, isec, nt = search_log_for_time(f)
            if ts:
                train_sec = ts
                if isec: infer_sec = isec
                notes = nt
                break
            elif isec and not infer_sec:
                infer_sec = isec

    # Fallback: search all *.log files (maxdepth 1) if still no time
    if not train_sec:
        all_logs_result = run(f'find "{path}" -maxdepth 1 -type f -name "*.log" 2>/dev/null', timeout=15)
        all_logs = [f for f in all_logs_result.split("\n") if f and f != "TIMEOUT"] if all_logs_result else []
        # Sort by size descending (larger logs more likely to have data)
        log_sizes = []
        for f in all_logs:
            if f in [mf for mf in meta_files]:
                continue  # already searched
            try:
                sz = os.path.getsize(f)
                log_sizes.append((sz, f))
            except:
                pass
        log_sizes.sort(reverse=True)
        for sz, f in log_sizes[:5]:  # check top 5 by size
            ts, isec, nt = search_log_for_time(f)
            if ts:
                train_sec = ts
                if isec: infer_sec = isec
                notes = nt
                break
            elif isec and not infer_sec:
                infer_sec = isec

    # Final fallback: compute from started=/finished= or START=/END= timestamp pairs
    if not train_sec:
        from datetime import datetime
        ts_patterns = [
            (re.compile(r'started\s*=\s*(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2})', re.IGNORECASE),
             re.compile(r'finished\s*=\s*(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2})', re.IGNORECASE)),
            (re.compile(r'START\s*=\s*(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2})', re.IGNORECASE),
             re.compile(r'END\s*=\s*(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2})', re.IGNORECASE)),
            (re.compile(r'===\s*START\s+(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2})', re.IGNORECASE),
             re.compile(r'===\s*END\s+(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2})', re.IGNORECASE)),
        ]
        all_logs_result2 = run(f'find "{path}" -maxdepth 1 -type f -name "*.log" 2>/dev/null', timeout=15)
        all_logs2 = [f for f in all_logs_result2.split("\n") if f and f != "TIMEOUT"] if all_logs_result2 else []
        for f in all_logs2 + meta_files:
            try:
                with open(f, 'r', errors='replace') as fh:
                    content = fh.read()
                for start_p, end_p in ts_patterns:
                    sm = start_p.search(content)
                    em = end_p.search(content)
                    if sm and em:
                        try:
                            t1 = datetime.fromisoformat(sm.group(1))
                            t2 = datetime.fromisoformat(em.group(1))
                            delta = (t2 - t1).total_seconds()
                            if delta > 0:
                                train_sec = f"{delta:.1f}"
                                notes = f"ts:{os.path.basename(f)}"
                                break
                        except:
                            pass
                if train_sec:
                    break
            except:
                pass

    ckpt_count = run(f'find "{path}" -maxdepth 4 -type f \\( -name "*.ckpt" -o -name "*.pt" -o -name "*.pth" -o -name "*.safetensors" \\) 2>/dev/null | wc -l', timeout=30)
    image_count = run(f'find "{path}" -maxdepth 4 -type f \\( -name "*.png" -o -name "*.jpg" -o -name "*.jpeg" \\) 2>/dev/null | wc -l', timeout=30)
    if ckpt_count == "TIMEOUT": ckpt_count = "scan_timeout"
    if image_count == "TIMEOUT": image_count = "scan_timeout"

    print(f"{name}\t{path}\t{mtime}\t{size_human}\t{train_sec}\t{human(train_sec)}\t{infer_sec}\t{ckpt_count}\t{image_count}\t{dataset}\t{notes}")

for path in sys.argv[1:]:
    probe(path)
