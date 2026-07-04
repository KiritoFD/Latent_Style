#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Scan local experiment directories and extract training/eval timing info."""
import csv
import json
import math
import os
import re
from datetime import datetime
from pathlib import Path

BASE = Path(r"g:\GitHub\Latent_Style\SchrodingerBridge\exp\exp_ours")
OUT = Path(r"g:\GitHub\Latent_Style\SchrodingerBridge\.trae\autoresearch\cleanup\local_timing.csv")

GROUPS = ["early", "phase4", "local_t"]

# task4_iter is a meta-dir whose sub-dirs are individual experiments
META_DIRS = {"early": ["task4_iter"]}

TS_REGEX = re.compile(r"(\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2})")


def list_experiments(group):
    """Return list of (relative_dir_name, absolute_path) for experiments in a group."""
    gpath = BASE / group
    exps = []
    meta = META_DIRS.get(group, [])
    for child in sorted(gpath.iterdir()):
        if not child.is_dir():
            continue
        if child.name in meta:
            # treat sub-dirs as experiments
            for sub in sorted(child.iterdir()):
                if sub.is_dir():
                    exps.append((f"{child.name}/{sub.name}", sub))
        else:
            exps.append((child.name, child))
    return exps


def find_train_log(exp_path):
    """Find train.log at top-level or src/. Fallback to any *.log at top-level."""
    for cand in [exp_path / "train.log", exp_path / "src" / "train.log"]:
        if cand.is_file():
            return cand
    # fallback: any *.log at top-level (not in src/)
    if exp_path.is_dir():
        for f in sorted(exp_path.iterdir()):
            if f.is_file() and f.suffix == ".log":
                return f
    return None


def parse_ts(line):
    """Extract first timestamp from a line. Returns datetime or None."""
    if not line:
        return None
    m = TS_REGEX.search(line)
    if not m:
        return None
    ts = m.group(1).replace("T", " ")
    try:
        return datetime.strptime(ts, "%Y-%m-%d %H:%M:%S")
    except ValueError:
        return None


def compute_train_duration(log_path):
    """Read first and last non-empty lines, extract timestamps, return minutes (float)."""
    try:
        with open(log_path, "r", encoding="utf-8", errors="replace") as f:
            lines = f.readlines()
    except Exception:
        return ""

    # find first line with a parseable timestamp
    first_ts = None
    for line in lines:
        ts = parse_ts(line)
        if ts:
            first_ts = ts
            break

    # find last line with a parseable timestamp
    last_ts = None
    for line in reversed(lines):
        ts = parse_ts(line)
        if ts:
            last_ts = ts
            break

    if first_ts and last_ts:
        delta = (last_ts - first_ts).total_seconds() / 60.0
        return round(delta, 2)
    return ""


def find_summary_json(exp_path):
    """Find the best summary.json under full_eval/.

    Preference order:
      1. plain 'full_eval' dir over 'full_eval_*' dirs
      2. direct child over nested (e.g. exp/<...>/full_eval)
      3. exclude paths containing 'src/' segment
      4. highest epoch number
      5. plain 'epoch_XXXX' over 'epoch_XXXX_full' (or other suffixes)
    Returns (relative_path_str, absolute_path) or (None, None).
    """
    if not exp_path.is_dir():
        return None, None

    # collect all full_eval* directories (direct + nested), excluding src/ paths
    candidates = []
    for p in exp_path.rglob("summary.json"):
        rel = p.relative_to(exp_path)
        parts = rel.parts
        # must contain a full_eval* component
        fe_idx = None
        for i, part in enumerate(parts):
            if part.startswith("full_eval"):
                fe_idx = i
                break
        if fe_idx is None:
            continue
        # exclude paths going through src/
        if "src" in parts[:fe_idx]:
            continue
        # the part after full_eval* should be epoch_*/summary.json
        if len(parts) < fe_idx + 3:
            continue
        if parts[-1] != "summary.json":
            continue
        fe_dir_name = parts[fe_idx]
        epoch_dir_name = parts[fe_idx + 1]
        m = re.match(r"epoch_(\d+)(.*)", epoch_dir_name)
        if not m:
            continue
        ep_num = int(m.group(1))
        suffix = m.group(2)
        is_plain_epoch = (suffix == "")
        is_direct = (fe_idx == 0)  # full_eval is a direct child of exp_path
        is_plain_fe = (fe_dir_name == "full_eval")
        # build relative path string
        rel_str = "/".join(parts)
        candidates.append((is_plain_fe, is_direct, -ep_num, 0 if is_plain_epoch else 1, rel_str, p))

    if not candidates:
        return None, None

    # sort: plain full_eval first, direct first, highest epoch first, plain epoch first
    candidates.sort(key=lambda x: (0 if x[0] else 1, 0 if x[1] else 1, x[2], x[3], x[4]))
    best = candidates[0]
    return best[4], best[5]


def _safe_round(val, ndigits):
    """Round to ndigits; return "" if NaN/None/invalid."""
    if val is None:
        return ""
    try:
        f = float(val)
    except (ValueError, TypeError):
        return ""
    if math.isnan(f) or math.isinf(f):
        return ""
    return round(f, ndigits)


def extract_summary_fields(sj_path):
    """Extract eval_duration_sec, clip_style, lpips, clip_content from summary.json."""
    try:
        with open(sj_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return "", "", "", ""

    # eval_duration_sec: try timings_sec.wall_total, then wall_time, eval_wall_seconds
    eval_dur = ""
    ts = data.get("timings_sec", {})
    if isinstance(ts, dict):
        for k in ("wall_total", "eval_total"):
            if k in ts:
                eval_dur = _safe_round(ts[k], 2)
                if eval_dur != "":
                    break
    if eval_dur == "":
        for k in ("wall_time", "eval_wall_seconds"):
            if k in data:
                eval_dur = _safe_round(data[k], 2)
                if eval_dur != "":
                    break

    # metrics from analysis.all_pairs_overview
    ao = data.get("analysis", {}).get("all_pairs_overview", {})
    if not isinstance(ao, dict):
        ao = {}

    def pick(keys):
        for k in keys:
            if k in ao:
                r = _safe_round(ao[k], 4)
                if r != "":
                    return r
        # also check top-level
        for k in keys:
            if k in data:
                r = _safe_round(data[k], 4)
                if r != "":
                    return r
        return ""

    clip_style = pick(["clip_style_score", "clip_style", "style_clip_score"])
    lpips = pick(["lpips", "lpips_score", "content_lpips"])
    clip_content = pick(["clip_content_score", "clip_content"])

    return eval_dur, clip_style, lpips, clip_content


def main():
    rows = []
    for group in GROUPS:
        for exp_name, exp_path in list_experiments(group):
            # train.log
            log_path = find_train_log(exp_path)
            train_log_exists = log_path is not None
            train_dur = ""
            if train_log_exists:
                train_dur = compute_train_duration(log_path)

            # summary.json
            eval_rel, sj_path = find_summary_json(exp_path)
            eval_dur, clip_style, lpips, clip_content = "", "", "", ""
            if sj_path:
                eval_dur, clip_style, lpips, clip_content = extract_summary_fields(sj_path)

            rows.append({
                "experiment_dir": exp_name,
                "group": group,
                "train_log_exists": str(train_log_exists).lower(),
                "train_duration_min": train_dur,
                "eval_summary_path": eval_rel if eval_rel else "",
                "eval_duration_sec": eval_dur,
                "clip_style": clip_style,
                "lpips": lpips,
                "clip_content": clip_content,
            })

    # write CSV
    fields = ["experiment_dir", "group", "train_log_exists", "train_duration_min",
              "eval_summary_path", "eval_duration_sec", "clip_style", "lpips", "clip_content"]
    with open(OUT, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

    # stats
    total = len(rows)
    with_train = sum(1 for r in rows if r["train_duration_min"] != "")
    with_eval = sum(1 for r in rows if r["eval_duration_sec"] != "")
    print(f"Total experiments: {total}")
    print(f"With train duration: {with_train}")
    print(f"With eval duration: {with_eval}")
    print(f"CSV written to: {OUT}")
    # breakdown by group
    for g in GROUPS:
        gr = [r for r in rows if r["group"] == g]
        print(f"  {g}: {len(gr)} experiments, "
              f"{sum(1 for r in gr if r['train_duration_min'] != '')} with train, "
              f"{sum(1 for r in gr if r['eval_duration_sec'] != '')} with eval")


if __name__ == "__main__":
    main()
