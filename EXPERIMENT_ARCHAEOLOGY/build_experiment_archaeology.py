from __future__ import annotations

import csv
import json
import os
import re
import subprocess
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SCAN_ROOTS = [
    ROOT,
    Path("I:/Github/Latent_Style"),
    Path("I:/Github/Latent_Style_TokenizerClean"),
]
OUT = ROOT / "EXPERIMENT_ARCHAEOLOGY"
BY_DATASET = OUT / "by_dataset"
SOURCES = OUT / "sources"
CLEANUP = OUT / "cleanup"
TIMING_DIR = ROOT / "SchrodingerBridge" / "docs" / "timing"
ROOT_MASTER = ROOT / "EXPERIMENT_ARCHAEOLOGY_MASTER.csv"

READABLE_EXTS = {
    ".csv",
    ".json",
    ".jsonl",
    ".md",
    ".txt",
    ".log",
    ".out",
    ".err",
    ".yaml",
    ".yml",
}
CHECKPOINT_EXTS = {".pt", ".pth", ".ckpt", ".model", ".pkl", ".npz"}
SKIP_DIR_NAMES = {".git", "__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache", "node_modules"}
MAX_READ_BYTES = 8 * 1024 * 1024
MAX_PARSE_CSV_BYTES = 2 * 1024 * 1024
SKIP_PARSE_SUBSTRINGS = {
    "Related_Works/repos/",
    "Related_Works/run_511/repos/",
    "SchrodingerBridge/archives/old_paper_workspaces/",
    "PaperOrchestra-0.2.0/",
}

METRIC_KEYS = {
    "clip_style",
    "content_lpips",
    "clip_content",
    "clip_dir",
    "lpips",
    "ssim_y",
    "edge_f1",
    "musiq",
    "maniqa",
    "dists_content",
    "hf_patch_kid",
    "plain_kid",
    "aggregate_art_fid",
    "artfid",
    "kid",
}
TIME_HEADER_HINTS = {
    "train_sec",
    "eval_sec",
    "elapsed_sec",
    "train_elapsed_sec",
    "infer_elapsed_sec",
    "train_wall",
    "train_time_label",
    "infer_sec",
    "eval_wall_seconds",
    "train_wall_seconds",
    "wall_total",
}

PARAMS_M = {
    "lancet": "3.91",
    "lbm": "3.91",
    "ours": "3.91",
    "samst": "6",
    "s2wat": "7",
    "stytr2": "48.34",
    "cast": "7.01",
    "aesfa": "3.22",
    "aespa": "24.20",
    "aespa-net": "24.20",
}


def rel(path: Path | str) -> str:
    p = Path(path)
    try:
        return str(p.resolve().relative_to(ROOT)).replace("\\", "/")
    except Exception:
        s = str(path).replace("\\", "/")
        root_s = str(ROOT).replace("\\", "/")
        return s.replace(root_s + "/", "")


def source_root_for(path: Path) -> str:
    try:
        resolved = path.resolve()
    except OSError:
        resolved = path
    for root in SCAN_ROOTS:
        if root.exists():
            try:
                resolved.relative_to(root.resolve())
                return str(root).replace("\\", "/")
            except ValueError:
                pass
    return str(path.anchor).replace("\\", "/")


def safe_name(name: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", name.strip())
    cleaned = re.sub(r"_+", "_", cleaned).strip("_")
    return cleaned or "unknown_dataset"


def file_mtime_iso(path: Path) -> str:
    try:
        return datetime.fromtimestamp(path.stat().st_mtime).isoformat(timespec="seconds")
    except OSError:
        return ""


def should_skip(path: Path) -> bool:
    return any(part in SKIP_DIR_NAMES for part in path.parts)


def read_text(path: Path) -> str | None:
    try:
        if path.stat().st_size > MAX_READ_BYTES:
            return None
        return path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None


def likely_parse_path(path: Path) -> bool:
    rp = rel(path)
    rp_lower = rp.lower()
    if any(part.lower() in rp_lower for part in SKIP_PARSE_SUBSTRINGS):
        return False
    name = path.name.lower()
    if name in {"metrics.csv", "aggregate_targetwise_artfid.csv"}:
        return False
    if path.suffix.lower() == ".csv" and path.stat().st_size > MAX_PARSE_CSV_BYTES:
        return False
    high_value_names = [
        "summary",
        "timing",
        "runtime",
        "train",
        "eval",
        "report",
        "inventory",
        "index",
        "master",
        "comparison",
        "manifest",
        "status",
        "log",
    ]
    high_value_dirs = [
        "/results/",
        "/docs/experiments/",
        "/full_eval/",
        "/runs/",
        "/outputs/",
        "/exp/",
        "/logs/",
        "/summary/",
        "/complete_750/",
    ]
    return any(token in name for token in high_value_names) or any(token in rp_lower for token in high_value_dirs)


def guess_dataset(text: str) -> str:
    t = text.lower().replace("\\", "/")
    rules = [
        ("distinct5_512", ["distinct5", "wikiart_distinct5"]),
        ("wikiart512_5style", ["wikiart512_5style", "wikiart512", "wikiart_512", "wikiart images 512", "wikiart_latents_512"]),
        ("legacy256_overfit50", ["legacy256", "overfit50", "protocol_a_800", "samam_wsl_mamba_256", "latent-256"]),
        ("strict_protocol_750", ["strict750", "strict_750", "complete_750", "protocol-750", "protocol750", "750"]),
        ("run511_5domain", ["run_511", "5-domain", "photo, hayao, monet, vangogh, cezanne"]),
        ("photo_monet_5x5", ["5x5", "photo_to_monet", "cut_5x5", "cyclegan_5x5", "sdturbo_5x5", "sdedit_multi"]),
        ("seedream_wikiart512", ["seedream", "modelscope_qwen_edit", "wikiart512_ema"]),
        ("s2wat", ["s2wat"]),
        ("unknown", []),
    ]
    for key, needles in rules:
        if any(n in t for n in needles):
            return key
    return "unknown"


def guess_method(text: str) -> str:
    t = text.lower().replace("\\", "/")
    rules = [
        ("LANCET/LBM", ["lancet", "lbm", "schrodingerbridge", "ours_epoch", "ours_", "s-add__"]),
        ("SaMST", ["samst"]),
        ("SaMAM", ["samam", "samam", "samam-main", "samam_wsl"]),
        ("S2WAT", ["s2wat"]),
        ("StyleID", ["styleid", "style-id"]),
        ("AdaIN", ["adain"]),
        ("StyTr2", ["stytr2", "stytr-2"]),
        ("CAST", ["cast"]),
        ("AesFA", ["aesfa"]),
        ("AesPA-Net", ["aespa"]),
        ("CUT", ["cut_"]),
        ("CycleGAN", ["cyclegan"]),
        ("SDEdit", ["sdedit"]),
        ("SD-Turbo", ["sdturbo", "sd-turbo"]),
        ("Seedream", ["seedream"]),
        ("IDT", ["idt"]),
    ]
    for method, needles in rules:
        if any(n in t for n in needles):
            return method
    return ""


def guess_resolution(text: str) -> str:
    t = text.lower()
    if "512" in t:
        return "512"
    if "256" in t:
        return "256"
    return ""


def parse_time_label(value: Any) -> tuple[str, str, str]:
    if value is None:
        return "", "", ""
    s = str(value).strip()
    if not s or s.lower() in {"nan", "none", "null"}:
        return "", "", ""
    m = re.search(r"(-?\d+(?:\.\d+)?)\s*(seconds?|secs?|sec|s|minutes?|mins?|min|m|hours?|hrs?|hr|h)\b", s, re.I)
    if m:
        return m.group(1), normalize_unit(m.group(2)), s
    if re.fullmatch(r"-?\d+(?:\.\d+)?", s):
        return s, "", s
    return "", "", s


def normalize_unit(unit: str) -> str:
    u = unit.lower()
    if u in {"seconds", "second", "secs", "sec", "s"}:
        return "s"
    if u in {"minutes", "minute", "mins", "min", "m"}:
        return "m"
    if u in {"hours", "hour", "hrs", "hr", "h"}:
        return "h"
    return unit


def params_for(method: str, note: str = "") -> str:
    t = f"{method} {note}".lower()
    for key, value in PARAMS_M.items():
        if key in t:
            return value
    return ""


def looks_like_per_image_csv(headers: list[str], row_count: int) -> bool:
    lower = {h.lower() for h in headers}
    imageish = {"source_image", "generated_image", "content_path", "style_path", "image", "filename"}
    return row_count > 100 and bool(lower & imageish)


def add_row(rows: list[dict[str, str]], **kwargs: Any) -> None:
    base = {
        "period": "",
        "dataset_or_setting": "",
        "dataset_key": "",
        "method": "",
        "variant_or_run": "",
        "scope": "",
        "resolution": "",
        "images": "",
        "clip_style": "",
        "content_lpips": "",
        "clip_content": "",
        "ssim_y": "",
        "edge_f1": "",
        "musiq": "",
        "hf_patch_kid": "",
        "plain_kid": "",
        "aggregate_art_fid": "",
        "train_time_value": "",
        "train_time_unit": "",
        "train_time_label": "",
        "infer_time_value": "",
        "infer_time_unit": "",
        "infer_time_label": "",
        "params_m": "",
        "hardware": "",
        "status": "",
        "source_path": "",
        "source_kind": "",
        "note": "",
    }
    for key, value in kwargs.items():
        if key in base:
            base[key] = "" if value is None else str(value)
    if not base["dataset_key"]:
        base["dataset_key"] = guess_dataset(" ".join([base["dataset_or_setting"], base["source_path"], base["variant_or_run"]]))
    if not base["dataset_or_setting"]:
        base["dataset_or_setting"] = base["dataset_key"]
    if not base["method"]:
        base["method"] = guess_method(" ".join([base["source_path"], base["variant_or_run"], base["note"]]))
    if not base["resolution"]:
        base["resolution"] = guess_resolution(" ".join([base["dataset_or_setting"], base["source_path"], base["variant_or_run"]]))
    if not base["params_m"]:
        base["params_m"] = params_for(base["method"], base["note"])
    rows.append(base)


def load_csv(path: Path) -> list[dict[str, str]]:
    try:
        with path.open("r", encoding="utf-8-sig", errors="replace", newline="") as f:
            return list(csv.DictReader(f))
    except Exception:
        return []


def parse_master_log(path: Path, rows: list[dict[str, str]]) -> None:
    for r in load_csv(path):
        train_value, train_unit, train_label = parse_time_label(r.get("train_wall"))
        add_row(
            rows,
            period=r.get("date", ""),
            dataset_or_setting=r.get("dataset", ""),
            dataset_key=r.get("dataset", ""),
            method=r.get("method", ""),
            variant_or_run=r.get("variant_or_point", ""),
            scope=r.get("scope", ""),
            resolution=guess_resolution(r.get("dataset", "")),
            clip_style=r.get("clip_style", ""),
            content_lpips=r.get("content_lpips", ""),
            aggregate_art_fid=r.get("artfid_or_aggregate_artfid", ""),
            train_time_value=train_value,
            train_time_unit=train_unit,
            train_time_label=train_label,
            hardware=r.get("hardware", ""),
            status=r.get("status", ""),
            source_path=r.get("evidence_path", "") or rel(path),
            source_kind="aaai2027_master_experiment_log",
            note=r.get("note", ""),
        )


def parse_artfid_points(path: Path, rows: list[dict[str, str]]) -> None:
    for r in load_csv(path):
        train_value, train_unit, train_label = parse_time_label(r.get("train_time_label"))
        add_row(
            rows,
            dataset_or_setting=r.get("dataset_title", "") or r.get("dataset", ""),
            dataset_key=r.get("dataset", ""),
            method=r.get("method", ""),
            variant_or_run=r.get("label", ""),
            scope=r.get("scope", ""),
            resolution=guess_resolution(r.get("dataset", "") + " " + r.get("dataset_title", "")),
            clip_style=r.get("clip_style", ""),
            content_lpips=r.get("content_lpips", ""),
            aggregate_art_fid=r.get("aggregate_art_fid", ""),
            train_time_value=train_value,
            train_time_unit=train_unit,
            train_time_label=train_label,
            source_path=r.get("summary_path", "") or rel(path),
            source_kind="artfid_comparison_points",
            note="ArtFID comparison point; train time preserved from train_time_label.",
        )


def parse_timing_summary(path: Path, rows: list[dict[str, str]]) -> None:
    for r in load_csv(path):
        run = r.get("run", "")
        train_v, train_u, train_label = parse_time_label(r.get("train_elapsed_sec"))
        infer_v, infer_u, infer_label = parse_time_label(r.get("infer_elapsed_sec"))
        if train_v and not train_u:
            train_u = "s"
            train_label = f"{train_v} s"
        if infer_v and not infer_u:
            infer_u = "s"
            infer_label = f"{infer_v} s"
        add_row(
            rows,
            dataset_or_setting="run_511 / strict protocol timing probes",
            dataset_key="run511_5domain",
            method=guess_method(run),
            variant_or_run=run,
            resolution="256",
            images=r.get("infer_images", ""),
            train_time_value=train_v,
            train_time_unit=train_u,
            train_time_label=train_label,
            infer_time_value=infer_v,
            infer_time_unit=infer_u,
            infer_time_label=infer_label,
            status="train=" + r.get("train_status", "") + "; infer=" + r.get("infer_status", ""),
            source_path=rel(path),
            source_kind="Related_Works timing_summary.csv",
            note=r.get("note", ""),
        )


def parse_repro_inventory(path: Path, rows: list[dict[str, str]]) -> None:
    for r in load_csv(path):
        if not any(r.get(k, "") for k in ["lpips", "clip_style", "train_sec", "infer_sec", "images", "checkpoints"]):
            continue
        train_v, train_u, train_label = parse_time_label(r.get("train_sec"))
        infer_v, infer_u, infer_label = parse_time_label(r.get("infer_sec"))
        if train_v and not train_u:
            train_u = "s"
            train_label = f"{train_v} s"
        if infer_v and not infer_u:
            infer_u = "s"
            infer_label = f"{infer_v} s"
        add_row(
            rows,
            dataset_or_setting=guess_dataset(" ".join([r.get("source", ""), r.get("run", ""), r.get("path", "")])),
            method=guess_method(r.get("run", "") + " " + r.get("path", "")),
            variant_or_run=r.get("run", ""),
            resolution=guess_resolution(r.get("path", "")),
            images=r.get("images", ""),
            clip_style=r.get("clip_style", ""),
            content_lpips=r.get("lpips", ""),
            clip_content=r.get("clip_content", ""),
            ssim_y=r.get("ssim_y", ""),
            edge_f1=r.get("edge_f1", ""),
            musiq=r.get("musiq", ""),
            hf_patch_kid=r.get("hf_patch_kid", ""),
            plain_kid=r.get("plain_kid", ""),
            train_time_value=train_v,
            train_time_unit=train_u,
            train_time_label=train_label,
            infer_time_value=infer_v,
            infer_time_unit=infer_u,
            infer_time_label=infer_label,
            status="train=" + r.get("train_status", "") + "; infer=" + r.get("infer_status", ""),
            source_path=r.get("path", "") or rel(path),
            source_kind="Related_Works repro_data_inventory.csv",
            note=f"inventory_source={r.get('source','')}; checkpoints={r.get('checkpoints','')}",
        )


def parse_baseline_suite_summary(path: Path, rows: list[dict[str, str]]) -> None:
    data = load_csv(path)
    if not data:
        return
    train = next((r for r in data if r.get("stage") == "train"), {})
    infer = next((r for r in data if r.get("stage") == "infer"), {})
    if not train and not infer:
        return
    method = guess_method(str(path))
    train_v, train_u, train_label = parse_time_label(train.get("elapsed_sec"))
    infer_v, infer_u, infer_label = parse_time_label(infer.get("elapsed_sec"))
    if train_v and not train_u:
        train_u = "s"
        train_label = f"{train_v} s"
    if infer_v and not infer_u:
        infer_u = "s"
        infer_label = f"{infer_v} s"
    add_row(
        rows,
        dataset_or_setting="run_511 review baseline suite full4g / strict 750",
        dataset_key="run511_5domain",
        method=method,
        variant_or_run=path.parent.name,
        resolution="256",
        images=infer.get("images", ""),
        train_time_value=train_v,
        train_time_unit=train_u,
        train_time_label=train_label,
        infer_time_value=infer_v,
        infer_time_unit=infer_u,
        infer_time_label=infer_label,
        status="train=" + train.get("status", "") + "; infer=" + infer.get("status", ""),
        source_path=rel(path),
        source_kind="review_baseline_suite_full4g summary.csv",
        note="End-to-end train/infer suite timing.",
    )


def flatten_summary_json(path: Path, obj: Any, rows: list[dict[str, str]]) -> None:
    if not isinstance(obj, dict):
        return
    overall = obj.get("overall") if isinstance(obj.get("overall"), dict) else obj
    timings = obj.get("timings_sec") if isinstance(obj.get("timings_sec"), dict) else {}
    if not any(k in overall for k in ["clip_style", "content_lpips", "clip_content", "lpips"]) and not timings:
        return
    infer_v = ""
    infer_u = ""
    infer_label = ""
    if "wall_total" in timings:
        infer_v = str(timings.get("wall_total"))
        infer_u = "s"
        infer_label = f"{infer_v} s internal timings_sec.wall_total"
    add_row(
        rows,
        period=file_mtime_iso(path)[:10],
        dataset_or_setting=guess_dataset(rel(path)),
        method=guess_method(rel(path)),
        variant_or_run=path.parent.parent.name if path.parent.name.startswith("epoch_") else path.parent.name,
        scope="full_eval" if "full_eval" in rel(path).lower() or path.name == "summary.json" else "",
        resolution=guess_resolution(rel(path)),
        images=obj.get("count", "") or obj.get("images", ""),
        clip_style=overall.get("clip_style", "") or overall.get("clip_style_all", ""),
        content_lpips=overall.get("content_lpips", "") or overall.get("content_lpips_all", "") or overall.get("lpips", ""),
        clip_content=overall.get("clip_content", ""),
        aggregate_art_fid=overall.get("aggregate_art_fid", "") or obj.get("aggregate_art_fid", ""),
        infer_time_value=infer_v,
        infer_time_unit=infer_u,
        infer_time_label=infer_label,
        source_path=rel(path),
        source_kind="summary.json",
        note="Parsed top-level/overall metrics; timings_sec.wall_total treated as full eval/inference wall if present.",
    )


def parse_generic_csv(path: Path, rows: list[dict[str, str]]) -> None:
    data = load_csv(path)
    if not data:
        return
    headers = [h for h in data[0].keys() if h is not None]
    hset = {h.lower() for h in headers}
    if looks_like_per_image_csv(headers, len(data)):
        return
    if not (hset & (METRIC_KEYS | TIME_HEADER_HINTS | {"method", "run", "dataset", "path", "images"})):
        return
    if path.name in {
        "aaai2027_master_experiment_log.csv",
        "artfid_comparison_points.csv",
        "timing_summary.csv",
        "repro_data_inventory.csv",
    }:
        return
    for r in data[:300]:
        if not any(str(r.get(k, "")).strip() for k in headers):
            continue
        train_raw = r.get("train_time_label") or r.get("train_wall") or r.get("train_sec") or r.get("train_elapsed_sec") or r.get("train_wall_seconds")
        infer_raw = r.get("infer_sec") or r.get("infer_elapsed_sec") or r.get("eval_sec") or r.get("eval_wall_seconds") or r.get("elapsed_sec")
        train_v, train_u, train_label = parse_time_label(train_raw)
        infer_v, infer_u, infer_label = parse_time_label(infer_raw)
        if train_v and not train_u and any(k in r for k in ["train_sec", "train_elapsed_sec", "train_wall_seconds"]):
            train_u = "s"
            train_label = f"{train_v} s"
        if infer_v and not infer_u and any(k in r for k in ["infer_sec", "infer_elapsed_sec", "eval_sec", "eval_wall_seconds", "elapsed_sec"]):
            infer_u = "s"
            infer_label = f"{infer_v} s"
        if not any([train_v, infer_v, r.get("clip_style"), r.get("lpips"), r.get("content_lpips"), r.get("images")]):
            continue
        method = r.get("method") or guess_method(" ".join([r.get("run", ""), r.get("path", ""), rel(path)]))
        add_row(
            rows,
            dataset_or_setting=r.get("dataset") or r.get("dataset_title") or guess_dataset(" ".join([rel(path), r.get("path", ""), r.get("run", "")])),
            method=method,
            variant_or_run=r.get("run") or r.get("label") or r.get("variant_or_point") or path.parent.name,
            scope=r.get("scope", ""),
            resolution=guess_resolution(" ".join([rel(path), r.get("dataset", ""), r.get("path", "")])),
            images=r.get("images", ""),
            clip_style=r.get("clip_style", ""),
            content_lpips=r.get("content_lpips") or r.get("lpips", ""),
            clip_content=r.get("clip_content", ""),
            ssim_y=r.get("ssim_y", ""),
            edge_f1=r.get("edge_f1", ""),
            musiq=r.get("musiq", ""),
            hf_patch_kid=r.get("hf_patch_kid", ""),
            plain_kid=r.get("plain_kid", ""),
            aggregate_art_fid=r.get("aggregate_art_fid") or r.get("artfid", ""),
            train_time_value=train_v,
            train_time_unit=train_u,
            train_time_label=train_label,
            infer_time_value=infer_v,
            infer_time_unit=infer_u,
            infer_time_label=infer_label,
            source_path=r.get("path") or rel(path),
            source_kind="generic_csv",
            note=f"Parsed from {rel(path)}",
        )


def parse_text_timing(path: Path, text: str, rows: list[dict[str, str]]) -> None:
    rel_path = rel(path)
    for match in re.finditer(r"(?:elapsed_sec|wall_seconds|EVAL_WALL_SECONDS|wall_total)\s*[:=]\s*(\d+(?:\.\d+)?)", text, re.I):
        value = match.group(1)
        add_row(
            rows,
            period=file_mtime_iso(path)[:10],
            dataset_or_setting=guess_dataset(rel_path + " " + text[:500]),
            method=guess_method(rel_path + " " + text[:500]),
            variant_or_run=path.parent.name,
            resolution=guess_resolution(rel_path + " " + text[:500]),
            infer_time_value=value,
            infer_time_unit="s",
            infer_time_label=f"{value} s ({match.group(0).split()[0]})",
            source_path=rel_path,
            source_kind="text_timing_regex",
            note="Timing regex hit in text/log; inspect source for exact stage.",
        )


def add_training_doc_rows(rows: list[dict[str, str]]) -> None:
    source = "SchrodingerBridge/archives/old_root_files/training_times_documentation.md"
    fixed = [
        ("LBM", "legacy256_overfit50 / 5-domain multi-style", "256", "309.9", "s", "", "", "3.91", "Ours 7 epochs; source table says 309.9 s and inference 85.4 s per 750 images.", "85.4", "s"),
        ("SaMST", "run_511 / 5-domain strict protocol", "256", "6768.7", "s", "", "", "6", "Estimated 100-epoch train from 1-epoch timing probe; inference 39.8 s per 750 images.", "39.8", "s"),
        ("S2WAT", "run_511 / 5-domain strict protocol", "256", "10600", "s", "", "", "7", "Estimated 2000-iteration train; inference not separately measured in this doc.", "", ""),
        ("StyTr2", "run_511 review baseline suite full4g", "256", "143.46", "s", "", "", "48.34", "Other-baselines table; inference 567.37 s per 750 images.", "567.37", "s"),
        ("CAST", "run_511 review baseline suite full4g", "256", "1759.80", "s", "", "", "7.01", "Other-baselines table; inference 75.47 s per 750 images.", "75.47", "s"),
        ("AesFA", "run_511 review baseline suite full4g", "256", "6607.60", "s", "", "", "3.22", "Other-baselines table; inference 40.26 s per 750 images.", "40.26", "s"),
        ("AesPA-Net", "run_511 review baseline suite full4g", "256", "366.30", "s", "", "", "24.20", "Other-baselines table; inference 345.28 s per 750 images.", "345.28", "s"),
        ("StyleID", "run_511 / strict 750", "256", "0", "training-free", "", "", "", "Training-free baseline; inference 603.32 s estimated/recorded.", "603.32", "s"),
        ("AdaIN v32k", "run_511 / strict 750", "256", "9220.39", "s", "", "", "", "AdaIN v32k 32k iterations; inference 9.28 s per 750 images.", "9.28", "s"),
        ("AdaIN vgg19", "run_511 / strict 750", "256", "262.78", "s", "", "", "", "AdaIN vgg19 2k iterations; inference 9.10 s per 750 images.", "9.10", "s"),
    ]
    for method, dataset, res, train_v, train_u, _a, _b, params, note, infer_v, infer_u in fixed:
        add_row(
            rows,
            dataset_or_setting=dataset,
            method=method,
            resolution=res,
            train_time_value=train_v,
            train_time_unit=train_u,
            train_time_label=(f"{train_v} {train_u}" if train_u != "training-free" else "training-free"),
            infer_time_value=infer_v,
            infer_time_unit=infer_u,
            infer_time_label=(f"{infer_v} {infer_u}" if infer_v else ""),
            params_m=params,
            source_path=source,
            source_kind="training_times_documentation.md",
            note=note,
        )


def scan_files() -> tuple[list[dict[str, str]], list[dict[str, str]], list[Path], list[dict[str, str]]]:
    evidence: list[dict[str, str]] = []
    directory_stats: dict[str, Counter] = defaultdict(Counter)
    text_files: list[Path] = []
    checkpoints: list[dict[str, str]] = []

    for scan_root in SCAN_ROOTS:
        if not scan_root.exists():
            continue
        for path in enumerate_candidate_files(scan_root):
            if should_skip(path):
                continue
            if path.is_dir():
                continue
            process_file(path, evidence, directory_stats, text_files, checkpoints)

    directory_rows = []
    for directory, counts in sorted(directory_stats.items()):
        if counts["evidence_files"] or counts["checkpoints"]:
            directory_rows.append(
                {
                    "directory": directory,
                    "source_root": counts.get("source_root", ""),
                    "dataset_guess": guess_dataset(directory),
                    "method_guess": guess_method(directory),
                    "files_seen": str(counts["files"]),
                    "evidence_files": str(counts["evidence_files"]),
                    "checkpoints": str(counts["checkpoints"]),
                    "note": "Experiment-like directory if evidence_files/checkpoints > 0.",
                }
            )
    return evidence, directory_rows, text_files, checkpoints


def enumerate_candidate_files(scan_root: Path) -> list[Path]:
    patterns = [f"*{ext}" for ext in sorted(READABLE_EXTS | CHECKPOINT_EXTS)]
    try:
        cmd = ["rg", "--files", "--hidden", "--no-ignore"]
        for pattern in patterns:
            cmd.extend(["-g", pattern])
        for skip in SKIP_DIR_NAMES:
            cmd.extend(["-g", f"!{skip}/**"])
        result = subprocess.run(
            cmd,
            cwd=scan_root,
            text=True,
            encoding="utf-8",
            errors="replace",
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            timeout=90,
            check=False,
        )
        if result.stdout:
            return [scan_root / line.strip() for line in result.stdout.splitlines() if line.strip()]
    except Exception:
        pass

    candidates: list[Path] = []
    wanted = READABLE_EXTS | CHECKPOINT_EXTS
    for dirpath, dirnames, filenames in os.walk(scan_root):
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIR_NAMES]
        for filename in filenames:
            p = Path(dirpath) / filename
            if p.suffix.lower() in wanted:
                candidates.append(p)
    return candidates


def process_file(
    path: Path,
    evidence: list[dict[str, str]],
    directory_stats: dict[str, Counter],
    text_files: list[Path],
    checkpoints: list[dict[str, str]],
) -> None:
        ext = path.suffix.lower()
        rel_path = rel(path)
        parent = rel(path.parent)
        directory_stats[parent]["files"] += 1
        directory_stats[parent]["source_root"] = source_root_for(path)
        if ext in CHECKPOINT_EXTS:
            size_mb = path.stat().st_size / (1024 * 1024)
            cleanup_class = "review_delete_candidate"
            lower = rel_path.lower()
            if any(n in lower for n in ["s-add__k-1_c-0_w-20_col-0", "local_wsl_wikiart512_hist_b32_e8", "aaai2027", "distinct5_512_20260602"]):
                cleanup_class = "likely_mainline_keep"
            elif any(n in lower for n in ["smoke", "tmp", "archive", "old_experiment_dirs", "run_511/outputs"]):
                cleanup_class = "likely_non_mainline_delete_candidate"
            checkpoints.append(
                {
                    "checkpoint_path": rel_path,
                    "source_root": source_root_for(path),
                    "size_mb": f"{size_mb:.3f}",
                    "modified": file_mtime_iso(path),
                    "dataset_guess": guess_dataset(rel_path),
                    "method_guess": guess_method(rel_path),
                    "cleanup_class": cleanup_class,
                    "note": "Manifest only; script does not delete checkpoints.",
                }
            )
            directory_stats[parent]["checkpoints"] += 1
        if ext in READABLE_EXTS:
            should_parse_content = likely_parse_path(path)
            if should_parse_content:
                text_files.append(path)
            directory_stats[parent]["evidence_files"] += 1
            text = read_text(path) if should_parse_content else None
            timing_hits = ""
            metric_hits = ""
            date_hits = ""
            if text is not None:
                timing_hits = str(len(re.findall(r"elapsed_sec|train_sec|infer_sec|wall_total|wall_seconds|EVAL_WALL_SECONDS|train_wall", text, re.I)))
                metric_hits = str(len(re.findall(r"clip_style|content_lpips|lpips|ssim_y|edge_f1|artfid|kid|musiq|maniqa", text, re.I)))
                dates = sorted(set(re.findall(r"20\d{2}[-_/]\d{2}[-_/]\d{2}", rel_path + "\n" + text[:20000])))
                date_hits = ";".join(dates[:12])
            evidence.append(
                {
                    "source_path": rel_path,
                    "source_root": source_root_for(path),
                    "extension": ext,
                    "size_bytes": str(path.stat().st_size),
                    "modified": file_mtime_iso(path),
                    "dataset_guess": guess_dataset(rel_path + ("\n" + text[:1000] if text else "")),
                    "method_guess": guess_method(rel_path + ("\n" + text[:1000] if text else "")),
                    "run_dir": parent,
                    "timing_hit_count": timing_hits,
                    "metric_hit_count": metric_hits,
                    "date_hits": date_hits,
                    "note": "indexed only" if text is None else "read and indexed",
                }
            )

def build_rows(text_files: list[Path]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    special_parsers = {
        ROOT / "SchrodingerBridge" / "docs" / "experiments" / "aaai2027_master_experiment_log.csv": parse_master_log,
        ROOT / "SchrodingerBridge" / "docs" / "experiments" / "comparison_20260602" / "artfid_comparison_points.csv": parse_artfid_points,
        ROOT / "Related_Works" / "results" / "metrics_summary" / "timing_summary.csv": parse_timing_summary,
        ROOT / "Related_Works" / "results" / "repro_data_inventory.csv": parse_repro_inventory,
    }
    for path, parser in special_parsers.items():
        if path.exists():
            parser(path, rows)
    add_training_doc_rows(rows)

    for path in text_files:
        rel_path = rel(path)
        if "review_baseline_suite_full4g" in rel_path and path.name == "summary.csv":
            parse_baseline_suite_summary(path, rows)
            continue
        if path.suffix.lower() == ".json" and path.name == "summary.json":
            text = read_text(path)
            if text:
                try:
                    flatten_summary_json(path, json.loads(text), rows)
                except Exception:
                    pass
        elif path.suffix.lower() == ".csv":
            parse_generic_csv(path, rows)
        elif path.suffix.lower() in {".md", ".log", ".txt", ".err", ".out"}:
            text = read_text(path)
            if text:
                parse_text_timing(path, text, rows)
    return dedupe_rows(rows)


def row_key(row: dict[str, str]) -> tuple[str, ...]:
    return (
        row.get("dataset_key", ""),
        row.get("method", ""),
        row.get("variant_or_run", ""),
        row.get("scope", ""),
        row.get("source_path", ""),
        row.get("train_time_label", ""),
        row.get("infer_time_label", ""),
        row.get("clip_style", ""),
        row.get("content_lpips", ""),
    )


def dedupe_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    seen = set()
    out = []
    for row in rows:
        key = row_key(row)
        if key in seen:
            continue
        seen.add(key)
        out.append(row)
    return out


def write_csv(path: Path, rows: list[dict[str, str]], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        keys = []
        for row in rows:
            for key in row:
                if key not in keys:
                    keys.append(key)
        fieldnames = keys
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_md(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def write_reports(rows: list[dict[str, str]], evidence: list[dict[str, str]], dirs: list[dict[str, str]], checkpoints: list[dict[str, str]]) -> None:
    by_dataset: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        by_dataset[row["dataset_key"] or "unknown"].append(row)
    for dataset, dataset_rows in sorted(by_dataset.items()):
        write_csv(BY_DATASET / f"{safe_name(dataset)}.csv", dataset_rows, MASTER_FIELDS)

    timing_rows = [
        row
        for row in rows
        if row.get("train_time_value") or row.get("infer_time_value") or row.get("train_time_label") or row.get("infer_time_label")
    ]
    write_csv(TIMING_DIR / "training_inference_timing_master.csv", timing_rows, MASTER_FIELDS)
    write_csv(OUT / "master_experiments.csv", rows, MASTER_FIELDS)
    write_csv(ROOT_MASTER, rows, MASTER_FIELDS)
    write_csv(SOURCES / "evidence_files.csv", evidence)
    write_csv(OUT / "directory_index.csv", dirs)
    write_csv(CLEANUP / "checkpoint_cleanup_candidates.csv", checkpoints)

    dataset_counts = Counter(row["dataset_key"] or "unknown" for row in rows)
    method_counts = Counter(row["method"] or "unknown" for row in rows)
    source_counts = Counter(row["source_kind"] or "unknown" for row in rows)
    ckpt_total_mb = sum(float(r["size_mb"]) for r in checkpoints)
    likely_delete = [r for r in checkpoints if "delete_candidate" in r["cleanup_class"]]

    readme = f"""# Experiment Archaeology Index

Generated by `EXPERIMENT_ARCHAEOLOGY/build_experiment_archaeology.py`.

This directory is a repository-wide archaeological inventory of experiment evidence. It keeps raw source paths in every row so that later table-building can trace a number back to its original log, summary, CSV, or markdown note.

## Main Files

- `../EXPERIMENT_ARCHAEOLOGY_MASTER.csv`: root-level normalized master CSV.
- `master_experiments.csv`: same normalized table inside this directory.
- `by_dataset/*.csv`: one CSV per detected dataset/setting.
- `directory_index.csv`: experiment-like directories with evidence/checkpoint counts.
- `sources/evidence_files.csv`: every readable evidence file that was indexed.
- `cleanup/checkpoint_cleanup_candidates.csv`: checkpoint cleanup manifest only. No checkpoint was deleted by this script.
- `../SchrodingerBridge/docs/timing/training_inference_timing_master.csv`: timing-focused subset.

## Coverage Summary

- Normalized experiment rows: {len(rows)}
- Timing rows: {len(timing_rows)}
- Indexed evidence files: {len(evidence)}
- Experiment-like directories: {len(dirs)}
- Checkpoint-like files found: {len(checkpoints)}
- Checkpoint-like bytes found: {ckpt_total_mb:.1f} MB
- Checkpoint delete candidates in manifest: {len(likely_delete)}

## Dataset Row Counts

{format_counter(dataset_counts)}

## Method Row Counts

{format_counter(method_counts)}

## Source Kinds

{format_counter(source_counts)}

## Important Limits

Remote results are represented only when their evidence has been copied back into this repository or referenced by an existing local report. The script does not SSH into remote machines.

Checkpoint cleanup is intentionally non-destructive in this pass. The manifest marks likely non-mainline checkpoints, but deletion should happen in a separate reviewed step after confirming mainline retention policy.
"""
    write_md(OUT / "README.md", readme)

    report = f"""# Repository Experiment Archaeology Report

## Method

The scan walks the full repository tree, excluding only source-control/cache folders such as `.git` and `__pycache__`. It indexes readable evidence extensions (`csv`, `json`, `jsonl`, `md`, `txt`, `log`, `out`, `err`, `yaml`, `yml`) and separately catalogs checkpoint-like payloads (`pt`, `pth`, `ckpt`, `model`, `pkl`, `npz`).

The normalized master table combines:

- Curated experiment ledgers under `SchrodingerBridge/docs/experiments`.
- Related-works reproduction ledgers under `Related_Works/results`.
- Baseline timing suite summaries under `Related_Works/run_511/outputs/review_baseline_suite_full4g`.
- Direct `summary.json` metric/timing files.
- Text/log timing regex hits such as `elapsed_sec`, `wall_total`, and `wall_seconds`.
- Historical timing documentation from `SchrodingerBridge/archives/old_root_files/training_times_documentation.md`.

## Timing Policy

Training units are preserved where possible. Labels like `1.2m`, `5.8h`, `training-free`, or `310s to reported point` are not converted into seconds in the master fields. Numeric seconds are used only when the source field itself was seconds, for example `elapsed_sec`, `train_sec`, or `timings_sec.wall_total`.

Inference timing means the source's recorded inference/evaluation/generation wall time. When the source is a `summary.json` with `timings_sec.wall_total`, this is stored as an inference/full-eval wall because the summary records generation plus evaluation work.

## Dataset Outputs

{format_dataset_files(by_dataset)}

## Cleanup Policy

`cleanup/checkpoint_cleanup_candidates.csv` is a deletion candidate manifest, not an executed cleanup. It classifies obvious smoke/archive/tmp checkpoints separately from paths that look mainline or AAAI-related. This preserves the requested data-first workflow while preventing accidental deletion of a checkpoint still needed by another active thread.

## Missing Or Weak Evidence

- Remote-only artifacts not copied into the repository cannot be fully enumerated from this local pass.
- Some rows come from summaries or markdown reports rather than raw per-run logs; `source_kind` and `source_path` identify that provenance.
- Per-image metric CSVs are indexed in `sources/evidence_files.csv` but intentionally not expanded into the experiment master to keep the master at run/experiment granularity.
- Checkpoint deletion has not been executed; the manifest is ready for review before a destructive cleanup commit.
"""
    write_md(OUT / "ARCHAEOLOGY_REPORT.md", report)

    timing_md = f"""# Training And Inference Timing Ledger

This timing directory contains the timing-focused subset of the repository-wide archaeology pass.

Main file: `training_inference_timing_master.csv`

Rows preserve the original unit semantics whenever the source did. Training labels such as `1.2m`, `5.8h`, `training-free`, and source-side estimates remain marked in `train_time_label`/`note` rather than being silently converted.

## Source Families

{format_counter(Counter(row["source_kind"] or "unknown" for row in timing_rows))}

## Known Gaps

- Remote jobs are included only when timing was copied back into this repo as CSV/JSON/MD/log evidence.
- Some historical baselines have estimated training time because the original full training log was not preserved; those rows are marked in `note`.
- Several methods have quality metrics but no train or inference timing in local evidence; they remain in the archaeology master but not necessarily in this timing subset.
"""
    write_md(TIMING_DIR / "README.md", timing_md)

    cleanup_md = f"""# Checkpoint Cleanup Candidate Manifest

No checkpoints were deleted.

The CSV in this directory lists {len(checkpoints)} checkpoint-like files totaling {ckpt_total_mb:.1f} MB. The `cleanup_class` column is conservative:

- `likely_mainline_keep`: path contains known mainline/AAAI/Distinct5/LBM evidence markers.
- `likely_non_mainline_delete_candidate`: path contains smoke, tmp, archive, old experiment, or run_511 output markers.
- `review_delete_candidate`: not recognized as mainline, requires manual review before deletion.

Use this file as the next-step review gate before any destructive cleanup.
"""
    write_md(CLEANUP / "README.md", cleanup_md)


def format_counter(counter: Counter) -> str:
    if not counter:
        return "- none"
    return "\n".join(f"- `{key}`: {value}" for key, value in counter.most_common(40))


def format_dataset_files(by_dataset: dict[str, list[dict[str, str]]]) -> str:
    lines = []
    for dataset, rows in sorted(by_dataset.items()):
        lines.append(f"- `by_dataset/{safe_name(dataset)}.csv`: {len(rows)} rows")
    return "\n".join(lines) if lines else "- none"


MASTER_FIELDS = [
    "period",
    "method",
    "dataset_or_setting",
    "dataset_key",
    "resolution",
    "variant_or_run",
    "scope",
    "images",
    "clip_style",
    "content_lpips",
    "clip_content",
    "ssim_y",
    "edge_f1",
    "musiq",
    "hf_patch_kid",
    "plain_kid",
    "aggregate_art_fid",
    "train_time_value",
    "train_time_unit",
    "train_time_label",
    "infer_time_value",
    "infer_time_unit",
    "infer_time_label",
    "params_m",
    "hardware",
    "status",
    "source_path",
    "source_kind",
    "note",
]


def main() -> None:
    for path in [OUT, BY_DATASET, SOURCES, CLEANUP, TIMING_DIR]:
        path.mkdir(parents=True, exist_ok=True)
    evidence, dirs, text_files, checkpoints = scan_files()
    rows = build_rows(text_files)
    write_reports(rows, evidence, dirs, checkpoints)
    print(json.dumps({
        "rows": len(rows),
        "evidence_files": len(evidence),
        "directories": len(dirs),
        "checkpoints": len(checkpoints),
        "root_master": rel(ROOT_MASTER),
        "timing_master": rel(TIMING_DIR / "training_inference_timing_master.csv"),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
