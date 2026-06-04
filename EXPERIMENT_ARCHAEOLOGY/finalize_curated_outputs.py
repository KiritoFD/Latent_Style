from __future__ import annotations

import csv
import json
import re
import shutil
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "EXPERIMENT_ARCHAEOLOGY"
FINAL_BY_DATASET = OUT / "final_by_dataset"
TIMING_DIR = ROOT / "SchrodingerBridge" / "docs" / "timing"
ROOT_MASTER = ROOT / "EXPERIMENT_ARCHAEOLOGY_MASTER.csv"

RAW_LOCAL = OUT / "_scratch_raw_scans" / "master_experiments.csv"
RAW_LOCAL_DIRS = OUT / "_scratch_raw_scans" / "directory_index.csv"
REMOTE = OUT / "remote_i_curated" / "remote_i_curated_experiments.csv"
REMOTE_TIMELINE = OUT / "remote_i_curated" / "remote_i_timeline.csv"


FIELDS = [
    "source_root",
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
    "validity_class",
    "note",
]


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", errors="replace", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, str]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def norm_path(row: dict[str, str]) -> str:
    return (row.get("source_path") or row.get("path") or "").replace("\\", "/")


def safe_name(s: str) -> str:
    name = re.sub(r"[^A-Za-z0-9_.-]+", "_", s or "unknown").strip("_")
    return name or "unknown"


def family_from_path(path: str, current: str) -> tuple[str, str]:
    p = path.replace("\\", "/")
    l = p.lower()
    if current and current != "unknown":
        return current, current
    rules = [
        ("distinct5_512", ["distinct5", "wikiart_distinct5"]),
        ("wikiart512_5style", ["wikiart512", "wikiart_latents_512", "wikiart_images_512"]),
        ("legacy256_overfit50", ["legacy256", "overfit50", "protocol_a_800", "latent-256"]),
        ("run511_5domain", ["run_511"]),
        ("cycle_nce", ["cycle-nce"]),
        ("schrodingerbridge_destructive_ablation", ["schrodingerbridge/ablation_destructive_7epoch"]),
        ("legacy_style_transfer_experiments", ["github/latent_style/experiments/", "latent_style/experiments/"]),
        ("schrodingerbridge_vae_backend", ["schrodingerbridge/exp/vae_backend", "vae_backend_256"]),
        ("schrodingerbridge_aaai2027", ["schrodingerbridge/exp/aaai2027", "configs/aaai2027"]),
        ("schrodingerbridge_weight_sweep", ["weight_sweep_40", "lambda_grid", "kinetic_sweep"]),
        ("schrodingerbridge_grid_search", ["grid_search_3epoch", "next_round_80", "orthogonal_phase_space_sweep"]),
        ("schrodingerbridge_frontier", ["schrodingerbridge/exp/frontier"]),
        ("schrodingerbridge_representation_probe", ["representation_probe", "representation_probes", "style_representation"]),
        ("schrodingerbridge_review_additional", ["review_additional_experiments"]),
        ("schrodingerbridge_exp_general", ["schrodingerbridge/exp/"]),
        ("schrodingerbridge_docs_experiments", ["schrodingerbridge/docs/experiments/"]),
        ("schrodingerbridge_root_legacy", ["schrodingerbridge/"]),
        ("related_works_baselines", ["related_works", "baseline_pipeline"]),
        ("seedream_wikiart512", ["seedream", "modelscope_qwen_edit"]),
        ("photo_monet_5x5", ["5x5", "sdedit_multi", "sdturbo", "cyclegan", "cut_5x5"]),
        ("paper_or_docs_only", ["aaai_submission", "paper_refine", "paperorchestra"]),
    ]
    for key, needles in rules:
        if any(n in l for n in needles):
            return key, key
    top = p.split("/")[0] if p else "unknown"
    if top and top not in {"I:", "G:"}:
        return f"path_family_{safe_name(top).lower()}", f"path_family_{safe_name(top).lower()}"
    return "unclassified_curated_experiments", "unclassified_curated_experiments"


def validity(row: dict[str, str]) -> str:
    path = norm_path(row).lower()
    source_kind = (row.get("source_kind") or "").lower()
    if any(row.get(k) for k in ["clip_style", "content_lpips", "clip_content", "aggregate_art_fid"]):
        return "metric_evidence"
    if row.get("train_time_value") or row.get("infer_time_value") or "timing" in source_kind:
        return "timing_evidence"
    if "summary" in source_kind or "summary.json" in path:
        return "summary_evidence"
    if "log" in source_kind or "/logs/" in path or path.endswith(".log"):
        return "log_evidence"
    return "indexed_curated_evidence"


def normalize_row(row: dict[str, str], source_root: str) -> dict[str, str]:
    out = {k: row.get(k, "") for k in FIELDS}
    out["source_root"] = row.get("source_root") or source_root
    key, setting = family_from_path(norm_path(row), row.get("dataset_key", ""))
    out["dataset_key"] = key
    out["dataset_or_setting"] = row.get("dataset_or_setting") if row.get("dataset_or_setting") and row.get("dataset_or_setting") != "unknown" else setting
    out["validity_class"] = validity(row)
    if not out["method"]:
        out["method"] = infer_method(norm_path(row))
    return out


def infer_method(path: str) -> str:
    l = path.lower()
    for method, needles in [
        ("LANCET/LBM", ["lancet", "lbm", "schrodingerbridge", "s-add__"]),
        ("SaMST", ["samst"]),
        ("SaMAM", ["samam"]),
        ("S2WAT", ["s2wat"]),
        ("StyleID", ["styleid"]),
        ("AdaIN", ["adain"]),
        ("StyTr2", ["stytr2"]),
        ("CAST", ["cast"]),
        ("AesFA", ["aesfa"]),
        ("AesPA-Net", ["aespa"]),
        ("CUT", ["cut_"]),
        ("CycleGAN", ["cyclegan"]),
        ("SDEdit", ["sdedit"]),
        ("Seedream", ["seedream"]),
        ("IDT", ["idt"]),
    ]:
        if any(n in l for n in needles):
            return method
    return ""


def keep_local(row: dict[str, str]) -> bool:
    path = norm_path(row)
    if path.startswith("EXPERIMENT_ARCHAEOLOGY"):
        return False
    lower = path.lower()
    if any(token in lower for token in ["/.codex_", "\\.codex_", ".codex_remote_patch", ".codex_compare", ".codex_amp_patch"]):
        return False
    if row.get("dataset_key") == "unknown" and not row.get("method") and not any(row.get(k) for k in ["clip_style", "content_lpips", "train_time_value", "infer_time_value"]):
        return False
    if row.get("source_kind") == "text_timing_regex" and not any(token in path.lower() for token in ["train", "eval", "generate", "watch", "summary"]):
        return False
    return True


def dedupe(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    seen = set()
    out = []
    for r in rows:
        key = (
            r.get("source_root", ""),
            r.get("dataset_key", ""),
            r.get("method", ""),
            r.get("variant_or_run", ""),
            r.get("scope", ""),
            r.get("source_path", ""),
            r.get("train_time_label", ""),
            r.get("infer_time_label", ""),
            r.get("clip_style", ""),
            r.get("content_lpips", ""),
        )
        if key in seen:
            continue
        seen.add(key)
        out.append(r)
    return out


def load_all_rows() -> list[dict[str, str]]:
    rows = []
    for r in read_csv(RAW_LOCAL):
        if keep_local(r):
            rows.append(normalize_row(r, "G:/GitHub/Latent_Style"))
    for r in read_csv(REMOTE):
        rows.append(normalize_row(r, "I:/"))
    return dedupe(rows)


def write_dataset_files(rows: list[dict[str, str]]) -> None:
    if FINAL_BY_DATASET.exists():
        shutil.rmtree(FINAL_BY_DATASET)
    grouped = defaultdict(list)
    for r in rows:
        grouped[r["dataset_key"]].append(r)
    for key, group in sorted(grouped.items()):
        write_csv(FINAL_BY_DATASET / f"{safe_name(key)}.csv", group, FIELDS)


def timeline_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    local_events = []
    for r in rows:
        if r.get("period"):
            local_events.append({
                "period": r.get("period", ""),
                "source_root": r.get("source_root", ""),
                "dataset_key": r.get("dataset_key", ""),
                "method": r.get("method", ""),
                "event_type": r.get("validity_class", ""),
                "path": r.get("source_path", ""),
                "time_hint": r.get("train_time_label") or r.get("infer_time_label", ""),
                "metric_hint": metric_hint(r),
                "note": r.get("note", ""),
            })
    for rr in read_csv(REMOTE_TIMELINE):
        local_events.append({
            "period": rr.get("period", ""),
            "source_root": rr.get("source_root", "I:/"),
            "dataset_key": family_from_path(rr.get("path", ""), rr.get("dataset_guess", ""))[0],
            "method": rr.get("method_guess", ""),
            "event_type": rr.get("event_type", "remote_log_file"),
            "path": rr.get("path", ""),
            "time_hint": rr.get("elapsed_sec_hint", ""),
            "metric_hint": "",
            "note": rr.get("note", ""),
        })
    return sorted(local_events, key=lambda x: x.get("period", ""))


def metric_hint(r: dict[str, str]) -> str:
    bits = []
    for key in ["clip_style", "content_lpips", "clip_content", "aggregate_art_fid"]:
        if r.get(key):
            bits.append(f"{key}={r[key]}")
    return "; ".join(bits)


def write_docs(rows: list[dict[str, str]], timeline: list[dict[str, str]]) -> None:
    dataset_counts = Counter(r["dataset_key"] for r in rows)
    method_counts = Counter(r["method"] or "unknown" for r in rows)
    root_counts = Counter(r["source_root"] for r in rows)
    validity_counts = Counter(r["validity_class"] for r in rows)
    timing_rows = [r for r in rows if r.get("train_time_value") or r.get("infer_time_value") or r.get("train_time_label") or r.get("infer_time_label")]
    local_delete = load_json(OUT / "cleanup" / "local_delete_summary.json")
    remote_delete = load_json(OUT / "remote_i_curated" / "remote_i_delete_summary.json")

    readme = [
        "# Experiment Archaeology",
        "",
        "This directory contains the curated local G and remote I experiment archaeology outputs.",
        "",
        "## Main Outputs",
        "",
        "- `../EXPERIMENT_ARCHAEOLOGY_MASTER.csv`: final root-level master CSV.",
        "- `final_master_experiments.csv`: same final master CSV inside this directory.",
        "- `final_by_dataset/*.csv`: one CSV per dataset/setting family.",
        "- `final_timeline.csv`: chronological experiment event index.",
        "- `EXPERIMENT_TIMELINE.md`: narrative timeline and experiment lineage.",
        "- `remote_i_curated/`: remote-side curated outputs generated after filtering and checkpoint cleanup.",
        "- `cleanup/local_deleted_checkpoints.csv`: local per-file deletion audit.",
        "- `remote_i_curated/remote_i_deleted_checkpoints.csv`: remote per-file deletion audit.",
        "- `SchrodingerBridge/docs/timing/training_inference_timing_master.csv`: timing-focused subset.",
        "",
        "## Counts",
        "",
        f"- Final experiment rows: {len(rows)}",
        f"- Timing rows: {len(timing_rows)}",
        f"- Timeline events: {len(timeline)}",
        f"- Source roots: {dict(root_counts)}",
        f"- Local deleted checkpoints: {local_delete.get('deleted_count', '')}, MB={local_delete.get('deleted_mb', '')}",
        f"- Remote deleted checkpoints: {remote_delete.get('deleted_count', '')}, MB={remote_delete.get('deleted_mb', '')}",
        "",
        "## Dataset Counts",
        "",
        *[f"- `{k}`: {v}" for k, v in dataset_counts.most_common()],
        "",
        "## Method Counts",
        "",
        *[f"- `{k}`: {v}" for k, v in method_counts.most_common(40)],
        "",
        "## Validity Classes",
        "",
        *[f"- `{k}`: {v}" for k, v in validity_counts.most_common()],
        "",
        "## Cleanup Rule",
        "",
        "Only explicitly non-mainline checkpoint candidates were deleted. Ambiguous `review_delete_candidate` files were retained.",
    ]
    (OUT / "README.md").write_text("\n".join(readme) + "\n", encoding="utf-8")

    periods = defaultdict(list)
    for ev in timeline:
        p = (ev.get("period") or "")[:10] or "unknown"
        periods[p].append(ev)
    lines = [
        "# Experiment Timeline And Lineage",
        "",
        "This log is derived from curated summaries, train/eval logs, timing rows, and remote I timeline records.",
        "",
    ]
    for period in sorted(periods):
        events = periods[period]
        ds = Counter(e["dataset_key"] for e in events)
        methods = Counter(e["method"] or "unknown" for e in events)
        lines.append(f"## {period}")
        lines.append("")
        lines.append(f"- Events: {len(events)}")
        lines.append(f"- Datasets/settings: {', '.join(f'{k}={v}' for k, v in ds.most_common(8))}")
        lines.append(f"- Methods: {', '.join(f'{k}={v}' for k, v in methods.most_common(8))}")
        for e in events[:25]:
            hint = e.get("time_hint") or e.get("metric_hint") or ""
            lines.append(f"- {e.get('event_type')}: {e.get('dataset_key')} / {e.get('method')} / {hint} / {e.get('path')}")
        if len(events) > 25:
            lines.append(f"- ... {len(events) - 25} more events in `final_timeline.csv`")
        lines.append("")
    (OUT / "EXPERIMENT_TIMELINE.md").write_text("\n".join(lines), encoding="utf-8")

    timing_md = [
        "# Training And Inference Timing",
        "",
        "This timing subset is regenerated from the final curated master table.",
        "",
        f"- Timing rows: {len(timing_rows)}",
        f"- Source roots: {dict(Counter(r['source_root'] for r in timing_rows))}",
        "",
        "Training units are preserved from the source where possible. Values such as `1.2m`, `5.8h`, or `training-free` are not silently converted to seconds.",
        "",
        "Known gaps: rows without timing remain in `EXPERIMENT_ARCHAEOLOGY_MASTER.csv`; remote-only artifacts not summarized by `summary.json` or logs are not expanded into metric rows.",
    ]
    TIMING_DIR.mkdir(parents=True, exist_ok=True)
    (TIMING_DIR / "README.md").write_text("\n".join(timing_md) + "\n", encoding="utf-8")


def load_json(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8-sig"))
    except Exception:
        return {}


def main() -> None:
    rows = load_all_rows()
    rows = sorted(rows, key=lambda r: (r.get("dataset_key", ""), r.get("period", ""), r.get("method", ""), r.get("source_path", "")))
    write_csv(OUT / "final_master_experiments.csv", rows, FIELDS)
    write_csv(ROOT_MASTER, rows, FIELDS)
    write_dataset_files(rows)
    tl = timeline_rows(rows)
    write_csv(OUT / "final_timeline.csv", tl, ["period", "source_root", "dataset_key", "method", "event_type", "path", "time_hint", "metric_hint", "note"])
    timing = [r for r in rows if r.get("train_time_value") or r.get("infer_time_value") or r.get("train_time_label") or r.get("infer_time_label")]
    write_csv(TIMING_DIR / "training_inference_timing_master.csv", timing, FIELDS)
    write_docs(rows, tl)
    print(json.dumps({"rows": len(rows), "timeline": len(tl), "timing": len(timing)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
