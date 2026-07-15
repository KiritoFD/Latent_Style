#!/usr/bin/env python3
"""Scan experiment directories and collect metadata.
Outputs JSON to stdout, progress to stderr."""
import os, json, subprocess, sys
from datetime import datetime

ROOTS = [
    "/mnt/i/Github/Latent_Style/Related_Works/baseline_pipeline/results",
    "/mnt/i/Github/Latent_Style/Related_Works/runs",
    "/mnt/i/Github/Latent_Style/exp",
    "/mnt/i/Github/Latent_Style/experiments",
    "/mnt/i/Github/Latent_Style/final_works",
]


def du_sh(path, timeout=30):
    try:
        out = subprocess.run(["du", "-sh", path], capture_output=True, text=True, timeout=timeout)
        return out.stdout.split()[0] if out.stdout else "?"
    except subprocess.TimeoutExpired:
        return "TIMEOUT"
    except Exception:
        return "ERR"


def find_summaries(path, maxdepth=3, max_count=10):
    summaries = []
    try:
        for root, dirs, files in os.walk(path):
            depth = root[len(path):].count(os.sep)
            if depth >= maxdepth:
                dirs[:] = []
                continue
            for f in files:
                if f == "summary.json" or f.endswith("_summary.json") or f == "metrics.csv":
                    summaries.append(os.path.join(root, f))
                    if len(summaries) >= max_count:
                        return summaries
    except Exception:
        pass
    return summaries


def extract_summary_fields(summary_path):
    fields = {}
    try:
        with open(summary_path) as f:
            data = json.load(f)
        keys_of_interest = [
            "wall_seconds","WALL_SECONDS","train_steps","epochs","dataset","config_name",
            "total_steps","training_time_sec","eval_seconds","num_steps","step","epoch",
            "runtime_seconds","elapsed_seconds","train_runtime_sec","WALL_TIME","wall_time",
            "training_wall_time","train_epochs","steps","global_step",
        ]
        if isinstance(data, dict):
            for k in keys_of_interest:
                if k in data:
                    fields[k] = data[k]
            if "config" in data and isinstance(data["config"], dict):
                for k in ["dataset","config_name","model_type","exp_name","exp","data"]:
                    if k in data["config"]:
                        fields[f"cfg.{k}"] = data["config"][k]
    except Exception as e:
        fields["_error"] = str(e)[:50]
    return fields


def count_files(path, extensions, maxdepth=4, max_count=100000):
    count = 0
    try:
        for root, dirs, files in os.walk(path):
            depth = root[len(path):].count(os.sep)
            if depth >= maxdepth:
                dirs[:] = []
                continue
            for f in files:
                for ext in extensions:
                    if f.endswith(ext):
                        count += 1
                        break
                if count > max_count:
                    return f">{max_count}"
    except Exception:
        return "ERR"
    return count


def get_mtime(path):
    try:
        return datetime.fromtimestamp(os.path.getmtime(path)).strftime("%Y-%m-%d %H:%M")
    except Exception:
        return "?"


def get_root_size(path):
    """Total size of root path itself, used for ROOTS summary."""
    return du_sh(path, timeout=120)


def main():
    results = []
    root_summaries = []
    progress_fp = open("/tmp/scan_progress.log", "w")
    def log(msg):
        progress_fp.write(msg + "\n")
        progress_fp.flush()
    for root_path in ROOTS:
        if not os.path.isdir(root_path):
            results.append({"root": root_path, "error": "NOT_FOUND"})
            continue
        root_size = get_root_size(root_path)
        root_summaries.append({"root": root_path, "size": root_size})
        try:
            entries = sorted(os.listdir(root_path))
        except Exception as e:
            results.append({"root": root_path, "error": str(e)})
            continue
        log(f"# Scanning {len(entries)} entries in {root_path} (size={root_size})")
        for i, entry in enumerate(entries):
            full = os.path.join(root_path, entry)
            if not os.path.isdir(full):
                continue
            if (i+1) % 10 == 0 or i == 0:
                log(f"  [{i+1}/{len(entries)}] {entry[:80]}")
            rec = {
                "root": os.path.basename(root_path),
                "name": entry,
                "mtime": get_mtime(full),
                "size": du_sh(full, timeout=30),
            }
            summaries = find_summaries(full, maxdepth=3, max_count=10)
            rec["summary_count"] = len(summaries)
            rec["has_metrics_csv"] = any(s.endswith("metrics.csv") for s in summaries)
            if summaries:
                summary_json = next((s for s in summaries if s.endswith("summary.json")), None)
                if summary_json:
                    rec["summary_fields"] = extract_summary_fields(summary_json)
            rec["ckpt_count"] = count_files(full, [".ckpt", ".pt", ".pth", ".safetensors"], maxdepth=4, max_count=100000)
            rec["img_count"] = count_files(full, [".png", ".jpg", ".jpeg"], maxdepth=3, max_count=100000)
            results.append(rec)

    output = {"root_summaries": root_summaries, "experiments": results}
    with open("/tmp/scan_results.json", "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"DONE: wrote /tmp/scan_results.json with {len(results)} entries")


if __name__ == "__main__":
    main()
