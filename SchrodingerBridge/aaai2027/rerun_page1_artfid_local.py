from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parent
WORKSPACE = ROOT.parent.parent
PAGE1_BUNDLE = ROOT / "page1_bundle"
DEFAULT_SEED_JSON_DIR = PAGE1_BUNDLE / "artfid_rerun_20260609"
DEFAULT_CURRENT_SUMMARY_CSV = PAGE1_BUNDLE / "page1_artfid_rerun_summary.csv"
DEFAULT_CURRENT_PANEL_CSV = PAGE1_BUNDLE / "page1_panel_points.csv"
DEFAULT_CURRENT_AUX_TABLE_CSV = ROOT / "distinct5_aux_artifact_table.csv"
TOOLS_DIR = ROOT.parent / "tools"

if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

import compute_targetwise_artfid_fast as artfid_fast


AUX_POINT_TO_LABEL = {
    "LBM-K e1": "LBM-K",
    "LBM-Knee e13": "LBM-Knee",
    "LBM-PS-v2 e13": "LBM-PS-v2",
    "SaMST e15": "SaMST e15",
    "Seedream-4.5": "Seedream-4.5",
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, str]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def resolve_path(raw: str) -> Path:
    candidate = Path(str(raw))
    if candidate.is_absolute():
        return candidate
    return WORKSPACE / candidate


def load_seed_payloads(
    *,
    seed_json_dir: Path,
    summary_csv: Path,
) -> list[dict[str, str]]:
    label_to_summary = {row["label"]: row for row in read_csv(summary_csv)}
    manifest_rows: list[dict[str, str]] = []
    for seed_json in sorted(seed_json_dir.glob("*.json")):
        payload = json.loads(seed_json.read_text(encoding="utf-8"))
        label = None
        for candidate_label, row in label_to_summary.items():
            if row.get("json") == seed_json.name:
                label = candidate_label
                train_time_label = row.get("train_time_label", "")
                break
        if label is None:
            raise KeyError(f"Could not map {seed_json.name} to a label from {summary_csv}")
        manifest_rows.append(
            {
                "label": label,
                "json": seed_json.name,
                "train_time_label": train_time_label,
                "generated_dir": str(resolve_path(payload["generated_dir"])),
                "metrics_path": str(resolve_path(payload["metrics_path"])),
                "source_root": str(resolve_path(payload["source_root"])),
                "target_root": str(resolve_path(payload["test_dir"])),
            }
        )
    return manifest_rows


def compute_rows(
    *,
    manifest_rows: list[dict[str, str]],
    out_json_dir: Path,
    batch_size: int,
    device: str,
    cache_dir: Path,
    scope_tag: str,
) -> list[dict[str, str]]:
    out_json_dir.mkdir(parents=True, exist_ok=True)
    feature_model = artfid_fast.load_artfid_feature_extractor(device=device, cache_dir=cache_dir)
    lpips_loss_fn = artfid_fast.load_artfid_lpips(device=device)
    ref_stats_cache: dict[Path, dict[str, dict[str, object]]] = {}

    refreshed_rows: list[dict[str, str]] = []
    total = len(manifest_rows)
    for idx, row in enumerate(manifest_rows, start=1):
        label = row["label"]
        target_root = Path(row["target_root"])
        if target_root not in ref_stats_cache:
            target_styles = [p.name for p in sorted(target_root.iterdir()) if p.is_dir()]
            ref_stats_cache[target_root] = artfid_fast.collect_reference_stats(
                target_styles,
                target_root=target_root,
                feature_model=feature_model,
                batch_size=batch_size,
                device=device,
            )

        started = time.perf_counter()
        payload = artfid_fast.compute_artfid_payload(
            images_dir=Path(row["generated_dir"]),
            metrics_csv=Path(row["metrics_path"]),
            target_root=target_root,
            source_root=Path(row["source_root"]),
            batch_size=batch_size,
            device=device,
            cache_dir=cache_dir,
            feature_model=feature_model,
            lpips_loss_fn=lpips_loss_fn,
            ref_stats=ref_stats_cache[target_root],
        )
        payload["wall_seconds"] = time.perf_counter() - started

        out_json = out_json_dir / row["json"]
        out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

        refreshed_rows.append(
            {
                "label": label,
                "json": row["json"],
                "train_time_label": row["train_time_label"],
                "scope": scope_tag,
                "aggregate_art_fid": str(payload["full"]["aggregate_art_fid"]),
                "aggregate_art_fid_fid": str(payload["full"]["aggregate_art_fid_fid"]),
                "aggregate_art_fid_content_lpips": str(payload["full"]["aggregate_art_fid_content_lpips"]),
                "count": str(payload["full"]["count"]),
                "transfer_aggregate_art_fid": str(payload["transfer"]["aggregate_art_fid"]),
                "transfer_aggregate_art_fid_fid": str(payload["transfer"]["aggregate_art_fid_fid"]),
                "transfer_aggregate_art_fid_content_lpips": str(payload["transfer"]["aggregate_art_fid_content_lpips"]),
                "transfer_count": str(payload["transfer"]["count"]),
                "wall_seconds": f"{float(payload['wall_seconds']):.6f}",
                "resolved_pair_count": str(payload["resolved_pair_count"]),
                "missing_pair_count": str(payload["missing_pair_count"]),
            }
        )
        print(
            f"[{idx}/{total}] {label}: "
            f"all={payload['full']['aggregate_art_fid']:.4f} "
            f"transfer={payload['transfer']['aggregate_art_fid']:.4f} "
            f"wall={payload['wall_seconds']:.2f}s"
        )
    return refreshed_rows


def update_panel_csv(
    *,
    template_panel_csv: Path,
    refreshed_rows: list[dict[str, str]],
    output_panel_csv: Path,
) -> None:
    panel_rows = read_csv(template_panel_csv)
    by_label = {row["label"]: row for row in refreshed_rows}
    for row in panel_rows:
        refreshed = by_label[row["label"]]
        row["train_time_label"] = refreshed["train_time_label"]
        row["tw_artfid_all"] = refreshed["aggregate_art_fid"]
        row["tw_artfid_transfer"] = refreshed["transfer_aggregate_art_fid"]
        row["artfid_json"] = refreshed["json"]
    write_csv(output_panel_csv, panel_rows, list(panel_rows[0].keys()))


def update_aux_csv(
    *,
    template_aux_csv: Path,
    refreshed_rows: list[dict[str, str]],
    output_aux_csv: Path,
) -> None:
    aux_rows = read_csv(template_aux_csv)
    by_label = {row["label"]: row for row in refreshed_rows}
    for row in aux_rows:
        label = AUX_POINT_TO_LABEL.get(row["point"])
        if label is None or label not in by_label:
            continue
        row["tw_artfid_all_down"] = f"{float(by_label[label]['aggregate_art_fid']):.4f}"
    write_csv(output_aux_csv, aux_rows, list(aux_rows[0].keys()))


def main() -> int:
    parser = argparse.ArgumentParser(description="Fresh local rerun for all page-1 ArtFID numbers.")
    parser.add_argument("--seed-json-dir", type=Path, default=DEFAULT_SEED_JSON_DIR)
    parser.add_argument("--current-summary-csv", type=Path, default=DEFAULT_CURRENT_SUMMARY_CSV)
    parser.add_argument("--current-panel-csv", type=Path, default=DEFAULT_CURRENT_PANEL_CSV)
    parser.add_argument("--current-aux-csv", type=Path, default=DEFAULT_CURRENT_AUX_TABLE_CSV)
    parser.add_argument("--tag", type=str, required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--cache-dir", type=Path, default=WORKSPACE / "eval_cache")
    parser.add_argument("--replace-current", action="store_true")
    args = parser.parse_args()

    tag = args.tag.strip()
    if not tag:
        raise ValueError("tag must be non-empty")

    scope_tag = f"paper_recompute_local_{tag}"
    manifest_rows = load_seed_payloads(seed_json_dir=args.seed_json_dir, summary_csv=args.current_summary_csv)

    manifest_csv = PAGE1_BUNDLE / f"page1_artfid_rerun_manifest_{tag}.csv"
    summary_csv = PAGE1_BUNDLE / f"page1_artfid_rerun_summary_{tag}.csv"
    panel_csv = PAGE1_BUNDLE / f"page1_panel_points_{tag}.csv"
    aux_csv = PAGE1_BUNDLE / f"distinct5_aux_artifact_table_{tag}.csv"
    out_json_dir = PAGE1_BUNDLE / f"artfid_rerun_local_{tag}"

    write_csv(manifest_csv, manifest_rows, list(manifest_rows[0].keys()))
    refreshed_rows = compute_rows(
        manifest_rows=manifest_rows,
        out_json_dir=out_json_dir,
        batch_size=max(1, int(args.batch_size)),
        device=args.device,
        cache_dir=args.cache_dir,
        scope_tag=scope_tag,
    )
    write_csv(summary_csv, refreshed_rows, list(refreshed_rows[0].keys()))
    update_panel_csv(
        template_panel_csv=args.current_panel_csv,
        refreshed_rows=refreshed_rows,
        output_panel_csv=panel_csv,
    )
    update_aux_csv(
        template_aux_csv=args.current_aux_csv,
        refreshed_rows=refreshed_rows,
        output_aux_csv=aux_csv,
    )

    if args.replace_current:
        write_csv(args.current_summary_csv, refreshed_rows, list(refreshed_rows[0].keys()))
        update_panel_csv(
            template_panel_csv=panel_csv,
            refreshed_rows=refreshed_rows,
            output_panel_csv=args.current_panel_csv,
        )
        update_aux_csv(
            template_aux_csv=aux_csv,
            refreshed_rows=refreshed_rows,
            output_aux_csv=args.current_aux_csv,
        )

    print("manifest_csv=", manifest_csv)
    print("summary_csv=", summary_csv)
    print("panel_csv=", panel_csv)
    print("aux_csv=", aux_csv)
    print("json_dir=", out_json_dir)
    if args.replace_current:
        print("current_summary_csv=", args.current_summary_csv)
        print("current_panel_csv=", args.current_panel_csv)
        print("current_aux_csv=", args.current_aux_csv)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
