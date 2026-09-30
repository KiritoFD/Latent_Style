from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path


TIMING_KEYS = (
    "wall_total",
    "eval_total",
    "lancet_generation",
    "generated_cpu_copy",
    "vae_decode",
    "eval_metrics_loop",
    "encode_inversion",
    "source_load_to_device",
)


def _epoch_int(name: str) -> int:
    digits = "".join(ch for ch in str(name) if ch.isdigit())
    return int(digits) if digits else -1


def _safe_float(value: object) -> float | None:
    try:
        return float(value)
    except Exception:
        return None


def _read_epoch_summary(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _collect_rows(eval_root: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for epoch_dir in sorted(eval_root.glob("epoch_*"), key=lambda p: _epoch_int(p.name)):
        summary_path = epoch_dir / "summary.json"
        if not summary_path.is_file():
            continue
        payload = _read_epoch_summary(summary_path)
        timings = payload.get("timings_sec") or {}
        row: dict[str, object] = {
            "epoch": epoch_dir.name,
            "summary_json": str(summary_path),
        }
        for key in TIMING_KEYS:
            row[key] = _safe_float(timings.get(key))
        rows.append(row)
    return rows


def _median(rows: list[dict[str, object]], key: str) -> float | None:
    values = [float(row[key]) for row in rows if isinstance(row.get(key), (int, float))]
    if not values:
        return None
    return float(statistics.median(values))


def _annotate(rows: list[dict[str, object]], *, anomaly_ratio: float) -> list[dict[str, object]]:
    medians = {key: _median(rows, key) for key in TIMING_KEYS}
    out: list[dict[str, object]] = []
    for row in rows:
        annotated = dict(row)
        flags: list[str] = []
        ratios: dict[str, float] = {}
        for key, median in medians.items():
            value = row.get(key)
            if median is None or not isinstance(value, (int, float)) or median <= 0:
                continue
            ratio = float(value) / float(median)
            ratios[f"{key}_ratio_vs_median"] = ratio
            if ratio >= float(anomaly_ratio):
                flags.append(key)
        annotated.update(ratios)
        annotated["anomaly_keys"] = "|".join(flags)
        annotated["is_anomalous"] = bool(flags)
        out.append(annotated)
    return out


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise RuntimeError("No timing rows to write.")
    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row.keys():
            if key in seen:
                continue
            seen.add(key)
            fieldnames.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path: Path, rows: list[dict[str, object]], *, anomaly_ratio: float) -> None:
    anomalies = [
        {
            "epoch": str(row.get("epoch", "")).strip(),
            "anomaly_keys": str(row.get("anomaly_keys", "")).strip(),
            "wall_total": row.get("wall_total"),
        }
        for row in rows
        if bool(row.get("is_anomalous"))
    ]
    payload = {
        "row_count": len(rows),
        "anomaly_ratio": float(anomaly_ratio),
        "anomaly_count": len(anomalies),
        "anomalies": anomalies,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit per-epoch round1 eval timings and flag anomalously slow checkpoints.")
    parser.add_argument("--eval-root", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--anomaly-ratio", type=float, default=1.5)
    args = parser.parse_args()

    eval_root = Path(args.eval_root).resolve()
    rows = _collect_rows(eval_root)
    if not rows:
        raise RuntimeError(f"No epoch summaries found under: {eval_root}")
    annotated = _annotate(rows, anomaly_ratio=float(args.anomaly_ratio))
    _write_csv(Path(args.output_csv).resolve(), annotated)
    _write_json(Path(args.output_json).resolve(), annotated, anomaly_ratio=float(args.anomaly_ratio))
    print(Path(args.output_csv).resolve())
    print(Path(args.output_json).resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
