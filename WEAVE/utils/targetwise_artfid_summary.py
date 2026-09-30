from __future__ import annotations

import json
from pathlib import Path


def build_targetwise_artfid_summary(matrix_breakdown: dict) -> dict | None:
    if not isinstance(matrix_breakdown, dict) or not matrix_breakdown:
        return None

    def _mean(values: list[float]) -> float | None:
        if not values:
            return None
        return float(sum(values) / len(values))

    def _build_scope(rows: list[dict]) -> dict:
        per_target = {}
        for row in rows:
            tgt = row["tgt_style"]
            bucket = per_target.setdefault(
                tgt,
                {
                    "art_fid": [],
                    "clip_style": [],
                    "content_lpips": [],
                    "count_pairs": 0,
                },
            )
            bucket["art_fid"].append(row["art_fid"])
            bucket["clip_style"].append(row["clip_style"])
            bucket["content_lpips"].append(row["content_lpips"])
            bucket["count_pairs"] += 1

        per_target_summary = {}
        target_means = []
        for tgt, bucket in sorted(per_target.items()):
            mean_art_fid = _mean(bucket["art_fid"])
            if mean_art_fid is not None:
                target_means.append(mean_art_fid)
            per_target_summary[tgt] = {
                "mean_art_fid": mean_art_fid,
                "mean_clip_style": _mean(bucket["clip_style"]),
                "mean_content_lpips": _mean(bucket["content_lpips"]),
                "count_pairs": int(bucket["count_pairs"]),
            }

        return {
            "count_pairs": int(len(rows)),
            "mean_art_fid": _mean([row["art_fid"] for row in rows]),
            "mean_of_target_means": _mean(target_means),
            "per_target": per_target_summary,
        }

    rows = []
    for src_style, tgt_map in matrix_breakdown.items():
        if not isinstance(tgt_map, dict):
            continue
        for tgt_style, stats in tgt_map.items():
            if not isinstance(stats, dict):
                continue
            art_fid = stats.get("art_fid")
            if art_fid is None:
                continue
            try:
                rows.append(
                    {
                        "src_style": str(src_style),
                        "tgt_style": str(tgt_style),
                        "art_fid": float(art_fid),
                        "clip_style": float(stats.get("clip_style", 0.0)),
                        "content_lpips": float(stats.get("content_lpips", 0.0)),
                    }
                )
            except Exception:
                continue

    if not rows:
        return None

    transfer_rows = [row for row in rows if row["src_style"] != row["tgt_style"]]
    identity_rows = [row for row in rows if row["src_style"] == row["tgt_style"]]

    return {
        "source": "summary.matrix_breakdown",
        "all_pairs": _build_scope(rows),
        "transfer_only": _build_scope(transfer_rows),
        "identity_only": _build_scope(identity_rows),
    }


def write_targetwise_artfid_summary(summary_json_path: str | Path, output_path: str | Path | None = None) -> Path | None:
    summary_json_path = Path(summary_json_path)
    if not summary_json_path.is_file():
        raise FileNotFoundError(summary_json_path)
    data = json.loads(summary_json_path.read_text(encoding="utf-8"))
    payload = build_targetwise_artfid_summary(data.get("matrix_breakdown", {}))
    if payload is None:
        return None
    target_path = Path(output_path) if output_path is not None else summary_json_path.with_name("aggregate_targetwise_artfid.json")
    target_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return target_path
