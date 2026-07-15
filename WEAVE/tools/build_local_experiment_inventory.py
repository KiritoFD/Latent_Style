"""Build a consolidated local experiment inventory and prune low-value artifacts.

This script scans the local experiment result surfaces that matter for the
current `Latent_Style` workspace:

- `SchrodingerBridge/exp/`
- `SchrodingerBridge/S-add__K-1_C-0_W-20_Col-0/`
- `Related_Works/baseline_pipeline/results/`

It writes a centralized inventory bundle under:

- `SchrodingerBridge/docs/experiments/inventory_20260603/`

Optional pruning is intentionally conservative:

- only local smoke / frozen exploratory surfaces are eligible
- only obvious non-data artifacts are deleted (images, grids, videos)
- logs, configs, summaries, metrics, checkpoints, and notes are preserved
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SCRIPT_PATH = Path(__file__).resolve()
SB_ROOT = SCRIPT_PATH.parents[1]
REPO_ROOT = SCRIPT_PATH.parents[2]
DOC_ROOT = SB_ROOT / "docs" / "experiments"
OUT_DIR = DOC_ROOT / "inventory_20260603"

LOCAL_SCAN_ROOTS = (
    ("sb_exp", SB_ROOT / "exp", "children"),
    ("sb_legacy_anchor", SB_ROOT / "S-add__K-1_C-0_W-20_Col-0", "self"),
    ("baseline_results", REPO_ROOT / "Related_Works" / "baseline_pipeline" / "results", "children"),
)

OUT_CSV = OUT_DIR / "local_experiment_inventory_20260603.csv"
OUT_JSON = OUT_DIR / "local_experiment_inventory_20260603.json"
OUT_MD = OUT_DIR / "README.md"
OUT_PRUNE_JSON = OUT_DIR / "local_prune_manifest_20260603.json"

CHECKPOINT_EXTS = {".pt", ".pth", ".ckpt", ".model", ".safetensors"}
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif", ".tif", ".tiff", ".svg"}
VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
KEEP_DATA_EXTS = {
    ".json",
    ".jsonl",
    ".csv",
    ".md",
    ".txt",
    ".log",
    ".yaml",
    ".yml",
    ".pt",
    ".pth",
    ".ckpt",
    ".model",
    ".safetensors",
    ".sh",
    ".ps1",
    ".bat",
}
PRUNE_EXTS = IMAGE_EXTS | VIDEO_EXTS

EXP_ACTIVE_NAMES = {
    "ablation_destructive_7epoch",
    "kinetic_sweep",
    "weight_sweep_40",
    "orth12",
    "legacy",
    "review_additional_experiments",
    "timing_20260601",
    "timing_20260602",
}
EXP_SUPPORT_NAMES = {
    "analysis",
    "configs",
    "diagnostics",
    "frontier",
    "inference",
    "paper",
    "runs",
    "scripts",
    "vae_backend",
    "video",
    "wikiart_512_encode_logs",
    "wikiart_512_transfer_logs",
}
EXP_FROZEN_PREFIXES = (
    "local_wsl_wikiart512_",
    "style_representation_",
    "tokenizer_",
    "fisher_",
    "memory_",
    "style_memory_",
    "router_",
    "typed_",
)
EXP_SMOKE_PREFIXES = ("_smoke_",)
BASELINE_SMOKE_TOKENS = ("smoke", "probe", "dry_", "rpm_probe", "channel_diag")
BASELINE_PAPER_TOKENS = (
    "samam_distinct5",
    "samst_distinct5_512_real_b2_e15_20260602",
    "samam_wsl_mamba_512_scratch_clean_silent_b1_20k",
    "samam_wsl_mamba_256_formal_750_eval",
    "samam_wsl_mamba_b2_15ep_15000",
    "convergence_summary_20260601",
    "timing_20260601",
    "timing_20260602",
    "samst",
)
BASELINE_EXTERNAL_TOKENS = (
    "seedream",
    "styleid",
    "s2wat",
    "cut",
    "sdturbo",
    "sdedit",
    "agnes_i2i",
    "modelscope_qwen_edit",
    "hf_",
    "local_instruct_pix2pix_probe",
    "z_image_turbo_smoke",
)
KNOWN_DISTINCT5_TOKENS = (
    "distinct5",
    "early_renaissance",
    "impressionism",
    "minimalism",
    "rococo",
    "ukiyo_e",
)
KNOWN_WIKIART512_TOKENS = (
    "wikiart512",
    "expressionism",
    "post_impressionism",
    "realism",
    "symbolism",
)
KNOWN_LEGACY256_TOKENS = (
    "protocol_a_800",
    "overfit50",
    "hayao",
    "cezanne",
    "vangogh",
    "photo",
    "monet",
    "legacy256",
    "256",
    "s-add__k-1_c-0_w-20_col-0",
)

CSV_FIELDS = [
    "scope",
    "root_family",
    "experiment_name",
    "root_path",
    "classification",
    "dataset_guess",
    "paper_usable",
    "prune_candidate",
    "checkpoint_count",
    "summary_count",
    "metrics_count",
    "artfid_count",
    "curve_metrics_count",
    "config_count",
    "log_count",
    "image_artifact_count",
    "total_size_mb",
    "image_artifact_size_mb",
    "latest_summary_path",
    "latest_summary_clip_style",
    "latest_summary_content_lpips",
    "best_clip_style",
    "best_content_lpips",
    "latest_artfid_path",
    "latest_artfid_full",
    "latest_artfid_transfer",
    "latest_log_path",
    "note",
]


@dataclass
class SummaryMetrics:
    clip_style: float | None = None
    content_lpips: float | None = None


@dataclass
class ArtFIDMetrics:
    full: float | None = None
    transfer: float | None = None


def _rel(path: Path) -> str:
    return str(path)


def _mb(num_bytes: int) -> float:
    return round(num_bytes / (1024.0 * 1024.0), 3)


def _safe_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _load_json(path: Path) -> dict[str, Any] | None:
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def _guess_dataset(path: Path) -> str:
    lowered = str(path).lower()
    if any(token in lowered for token in KNOWN_DISTINCT5_TOKENS):
        return "distinct5_512"
    if any(token in lowered for token in KNOWN_WIKIART512_TOKENS):
        return "wikiart512_5style"
    if any(token in lowered for token in KNOWN_LEGACY256_TOKENS):
        return "legacy256_overfit50"
    if "wikiart" in lowered and "512" in lowered:
        return "wikiart512"
    return "unknown"


def _guess_dataset_from_summary(path: Path) -> str:
    payload = _load_json(path)
    if not payload:
        return "unknown"
    matrix = payload.get("matrix_breakdown")
    if not isinstance(matrix, dict):
        return "unknown"
    keys = {str(key).lower() for key in matrix.keys()}
    if {"early_renaissance", "impressionism", "minimalism", "rococo", "ukiyo_e"} <= keys:
        return "distinct5_512"
    if {"expressionism", "impressionism", "post_impressionism", "realism", "symbolism"} <= keys:
        return "wikiart512_5style"
    if {"photo", "hayao", "monet", "vangogh", "cezanne"} <= keys:
        return "legacy256_overfit50"
    return "unknown"


def _guess_dataset_from_artfid(path: Path) -> str:
    payload = _load_json(path)
    if not payload:
        return "unknown"
    test_dir = payload.get("test_dir")
    if not isinstance(test_dir, str):
        return "unknown"
    return _guess_dataset(Path(test_dir))


def _classify_exp(name: str) -> tuple[str, str, str]:
    lowered = name.lower()
    if lowered.startswith("aaai2027_"):
        return ("formal_remote_packet", "yes", "no")
    if lowered.startswith("distinct5_512_ema_variant_") and lowered.endswith("_remote"):
        return ("active_paper_evidence", "yes", "no")
    if name in EXP_ACTIVE_NAMES:
        return ("active_paper_evidence", "conditional", "no")
    if name in EXP_SUPPORT_NAMES:
        return ("runtime_support", "no", "no")
    if name in {"probes_20260601", "reference_memory_generation_probe_full", "remote_factorized_tokenizer_pull", "seedream_distill_adapter"}:
        return ("local_exploratory_frozen", "no", "yes")
    if lowered.startswith(EXP_SMOKE_PREFIXES) or "smoke" in lowered:
        return ("local_smoke", "no", "yes")
    if lowered.startswith(EXP_FROZEN_PREFIXES):
        return ("local_exploratory_frozen", "no", "yes")
    return ("unclassified_review", "conditional", "no")


def _classify_baseline(name: str) -> tuple[str, str, str]:
    lowered = name.lower()
    if any(token in lowered for token in BASELINE_SMOKE_TOKENS):
        return ("local_smoke", "no", "yes")
    if any(token in lowered for token in BASELINE_EXTERNAL_TOKENS):
        return ("external_baseline_or_probe", "conditional", "no")
    if any(token in lowered for token in BASELINE_PAPER_TOKENS):
        return ("baseline_paper_or_historical", "yes", "no")
    return ("baseline_misc", "conditional", "no")


def _classify_anchor(name: str) -> tuple[str, str, str]:
    if name == "S-add__K-1_C-0_W-20_Col-0":
        return ("historical_anchor", "yes", "no")
    return ("anchor_misc", "conditional", "no")


def _parse_summary_metrics(path: Path) -> SummaryMetrics:
    payload = _load_json(path)
    if not payload:
        return SummaryMetrics()
    analysis = payload.get("analysis")
    if isinstance(analysis, dict):
        for key in ("all_pairs_overview", "overall", "transfer_excluding_identity", "style_transfer_ability"):
            node = analysis.get(key)
            if isinstance(node, dict):
                clip_style = _safe_float(node.get("clip_style"))
                content_lpips = _safe_float(node.get("content_lpips"))
                if clip_style is not None or content_lpips is not None:
                    return SummaryMetrics(clip_style=clip_style, content_lpips=content_lpips)
    clip_style = _safe_float(payload.get("clip_style"))
    content_lpips = _safe_float(payload.get("content_lpips"))
    return SummaryMetrics(clip_style=clip_style, content_lpips=content_lpips)


def _parse_artfid_metrics(path: Path) -> ArtFIDMetrics:
    payload = _load_json(path)
    if not payload:
        return ArtFIDMetrics()
    full = None
    transfer = None
    full_node = payload.get("full")
    if isinstance(full_node, dict):
        full = _safe_float(full_node.get("aggregate_art_fid"))
    transfer_node = payload.get("transfer")
    if isinstance(transfer_node, dict):
        transfer = _safe_float(transfer_node.get("aggregate_art_fid"))
    if full is None:
        full_node = payload.get("all_pairs")
        if isinstance(full_node, dict):
            full = _safe_float(full_node.get("mean_art_fid"))
            if full is None:
                full = _safe_float(full_node.get("mean_of_target_means"))
    if transfer is None:
        transfer_node = payload.get("transfer_only")
        if isinstance(transfer_node, dict):
            transfer = _safe_float(transfer_node.get("mean_art_fid"))
            if transfer is None:
                transfer = _safe_float(transfer_node.get("mean_of_target_means"))
    return ArtFIDMetrics(full=full, transfer=transfer)


def _latest_by_mtime(paths: list[Path]) -> Path | None:
    if not paths:
        return None
    return max(paths, key=lambda p: p.stat().st_mtime)


def _should_prune_file(path: Path) -> bool:
    lowered = path.name.lower()
    if path.suffix.lower() not in PRUNE_EXTS:
        return False
    if lowered in {"paper_aaai2026.pdf"}:
        return False
    return True


def _iter_experiment_roots() -> list[tuple[str, Path]]:
    roots: list[tuple[str, Path]] = []
    for family, root, mode in LOCAL_SCAN_ROOTS:
        if not root.exists():
            continue
        if mode == "self":
            roots.append((family, root))
            continue
        for child in sorted(root.iterdir()):
            if child.is_dir():
                roots.append((family, child))
    return roots


def scan_one(root_family: str, root: Path) -> dict[str, Any]:
    name = root.name
    if root_family == "sb_exp":
        classification, paper_usable, prune_candidate = _classify_exp(name)
    elif root_family == "baseline_results":
        classification, paper_usable, prune_candidate = _classify_baseline(name)
    else:
        classification, paper_usable, prune_candidate = _classify_anchor(name)

    all_files = [p for p in root.rglob("*") if p.is_file()]
    checkpoint_files = [p for p in all_files if p.suffix.lower() in CHECKPOINT_EXTS]
    summary_files = [p for p in all_files if p.name == "summary.json"]
    metrics_files = [p for p in all_files if p.name == "metrics.csv"]
    artfid_files = [p for p in all_files if p.name == "aggregate_targetwise_artfid.json"]
    curve_metrics_files = [p for p in all_files if p.name in {"curve_metrics.csv", "sb_curve_metrics.csv"}]
    config_files = [p for p in all_files if p.name in {"config.json", "hparams.yaml", "hparams.yml", "train.yml"}]
    log_files = [
        p
        for p in all_files
        if p.suffix.lower() in {".log", ".txt"}
        or p.name.startswith("training_")
        or p.name in {"gpu.csv", "watch.log"}
    ]
    image_files = [p for p in all_files if p.suffix.lower() in IMAGE_EXTS | VIDEO_EXTS]

    total_bytes = sum(p.stat().st_size for p in all_files)
    image_bytes = sum(p.stat().st_size for p in image_files)

    latest_summary = _latest_by_mtime(summary_files)
    latest_artfid = _latest_by_mtime(artfid_files)
    latest_log = _latest_by_mtime(log_files)

    latest_summary_metrics = _parse_summary_metrics(latest_summary) if latest_summary else SummaryMetrics()
    latest_artfid_metrics = _parse_artfid_metrics(latest_artfid) if latest_artfid else ArtFIDMetrics()
    dataset_guess = _guess_dataset(root)
    if dataset_guess == "unknown" and latest_artfid:
        dataset_guess = _guess_dataset_from_artfid(latest_artfid)
    if dataset_guess == "unknown" and latest_summary:
        dataset_guess = _guess_dataset_from_summary(latest_summary)

    best_clip_style = None
    best_content_lpips = None
    for path in summary_files:
        metrics = _parse_summary_metrics(path)
        if metrics.clip_style is not None:
            best_clip_style = metrics.clip_style if best_clip_style is None else max(best_clip_style, metrics.clip_style)
        if metrics.content_lpips is not None:
            best_content_lpips = (
                metrics.content_lpips if best_content_lpips is None else min(best_content_lpips, metrics.content_lpips)
            )

    note_bits: list[str] = []
    if latest_summary is None:
        note_bits.append("no summary.json")
    if latest_artfid is None:
        note_bits.append("no aggregate_targetwise_artfid.json")
    if classification in {"local_smoke", "local_exploratory_frozen"}:
        note_bits.append("eligible for non-data artifact pruning")
    if root_family == "baseline_results" and "distinct5" in name.lower() and "samst" in name.lower():
        note_bits.append("distinct5 baseline run")
    if root_family == "sb_exp" and classification == "formal_remote_packet":
        note_bits.append("formal aaai2027 packet")

    return {
        "scope": "local",
        "root_family": root_family,
        "experiment_name": name,
        "root_path": _rel(root),
        "classification": classification,
        "dataset_guess": dataset_guess,
        "paper_usable": paper_usable,
        "prune_candidate": prune_candidate,
        "checkpoint_count": len(checkpoint_files),
        "summary_count": len(summary_files),
        "metrics_count": len(metrics_files),
        "artfid_count": len(artfid_files),
        "curve_metrics_count": len(curve_metrics_files),
        "config_count": len(config_files),
        "log_count": len(log_files),
        "image_artifact_count": len(image_files),
        "total_size_mb": _mb(total_bytes),
        "image_artifact_size_mb": _mb(image_bytes),
        "latest_summary_path": _rel(latest_summary) if latest_summary else "",
        "latest_summary_clip_style": latest_summary_metrics.clip_style,
        "latest_summary_content_lpips": latest_summary_metrics.content_lpips,
        "best_clip_style": best_clip_style,
        "best_content_lpips": best_content_lpips,
        "latest_artfid_path": _rel(latest_artfid) if latest_artfid else "",
        "latest_artfid_full": latest_artfid_metrics.full,
        "latest_artfid_transfer": latest_artfid_metrics.transfer,
        "latest_log_path": _rel(latest_log) if latest_log else "",
        "note": "; ".join(note_bits),
    }


def scan_local() -> list[dict[str, Any]]:
    rows = [scan_one(root_family, root) for root_family, root in _iter_experiment_roots()]
    rows.sort(key=lambda row: (row["root_family"], row["classification"], row["experiment_name"]))
    return rows


def prune_local(rows: list[dict[str, Any]]) -> dict[str, Any]:
    deleted: list[dict[str, Any]] = []
    total_deleted_bytes = 0
    for row in rows:
        if row["prune_candidate"] != "yes":
            continue
        root = Path(row["root_path"])
        if not root.exists():
            continue
        for path in root.rglob("*"):
            if not path.is_file():
                continue
            if not _should_prune_file(path):
                continue
            size = path.stat().st_size
            path.unlink()
            total_deleted_bytes += size
            deleted.append(
                {
                    "experiment_name": row["experiment_name"],
                    "classification": row["classification"],
                    "deleted_path": _rel(path),
                    "size_bytes": size,
                    "size_mb": _mb(size),
                }
            )
    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "deleted_file_count": len(deleted),
        "deleted_total_mb": _mb(total_deleted_bytes),
        "deleted": deleted,
    }


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in CSV_FIELDS})


def write_json(rows: list[dict[str, Any]], prune_manifest: dict[str, Any] | None, path: Path) -> None:
    payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "scan_roots": [
            {"root_family": family, "path": _rel(root), "mode": mode}
            for family, root, mode in LOCAL_SCAN_ROOTS
        ],
        "row_count": len(rows),
        "rows": rows,
        "prune_manifest": prune_manifest,
    }
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=True, indent=2)


def write_markdown(rows: list[dict[str, Any]], prune_manifest: dict[str, Any] | None, path: Path) -> None:
    by_classification: dict[str, int] = {}
    by_family: dict[str, int] = {}
    for row in rows:
        by_classification[row["classification"]] = by_classification.get(row["classification"], 0) + 1
        by_family[row["root_family"]] = by_family.get(row["root_family"], 0) + 1

    interesting = [
        row
        for row in rows
        if row["paper_usable"] in {"yes", "conditional"}
        and (row["summary_count"] or row["artfid_count"] or row["checkpoint_count"])
    ]
    interesting.sort(
        key=lambda row: (
            0 if row["paper_usable"] == "yes" else 1,
            -(row["summary_count"] + row["artfid_count"] + row["checkpoint_count"]),
            row["experiment_name"],
        )
    )
    prune_rows = [row for row in rows if row["prune_candidate"] == "yes"]

    lines = [
        "# Local Experiment Inventory",
        "",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "Centralized outputs in this directory:",
        "",
        f"- `{OUT_CSV.name}`",
        f"- `{OUT_JSON.name}`",
        f"- `{OUT_PRUNE_JSON.name}`",
        "",
        "## Scan roots",
        "",
    ]
    for family, root, _mode in LOCAL_SCAN_ROOTS:
        lines.append(f"- `{family}`: `{root}`")
    lines.extend(["", "## Counts by root family", ""])
    for family, count in sorted(by_family.items()):
        lines.append(f"- `{family}`: {count}")
    lines.extend(["", "## Counts by classification", ""])
    for classification, count in sorted(by_classification.items()):
        lines.append(f"- `{classification}`: {count}")
    lines.extend(["", "## Paper-facing or review-worthy results", ""])
    for row in interesting[:20]:
        lines.append(
            "- "
            f"`{row['experiment_name']}` | `{row['root_family']}` | `{row['dataset_guess']}` | "
            f"class=`{row['classification']}` | ckpt={row['checkpoint_count']} | "
            f"summary={row['summary_count']} | artfid={row['artfid_count']} | "
            f"best_style={row['best_clip_style']} | best_lpips={row['best_content_lpips']}"
        )
    lines.extend(["", "## Local prune boundary", ""])
    lines.append(
        f"- prune-eligible directories: {len(prune_rows)} "
        "(only `local_smoke` / `local_exploratory_frozen`)"
    )
    if prune_manifest:
        lines.append(
            f"- deleted non-data artifacts: {prune_manifest['deleted_file_count']} files, "
            f"{prune_manifest['deleted_total_mb']} MB"
        )
    else:
        lines.append("- pruning not executed in this run")
    lines.extend(["", "## Notes", ""])
    lines.append("- This inventory keeps logs, configs, summaries, metrics, notes, and checkpoints.")
    lines.append("- Pruning deletes only obvious non-data artifacts such as grids, images, and videos.")
    lines.append("- Remote results are intentionally not included here; they belong in the separate remote inventory.")

    with path.open("w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--prune-low-value-artifacts",
        action="store_true",
        help="Delete only obvious non-data artifacts from local smoke/frozen exploratory directories.",
    )
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = scan_local()
    prune_manifest = None
    if args.prune_low_value_artifacts:
        prune_manifest = prune_local(rows)
        rows = scan_local()
        with OUT_PRUNE_JSON.open("w", encoding="utf-8") as f:
            json.dump(prune_manifest, f, ensure_ascii=True, indent=2)

    write_csv(rows, OUT_CSV)
    write_json(rows, prune_manifest, OUT_JSON)
    write_markdown(rows, prune_manifest, OUT_MD)


if __name__ == "__main__":
    main()
