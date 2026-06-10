from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _match_row(
    row: dict[str, str],
    *,
    epochs: set[str],
    reason_contains: list[str],
) -> bool:
    epoch = str(row.get("epoch", "")).strip()
    reason = str(row.get("reason", "")).strip()
    if epochs and epoch not in epochs:
        return False
    if reason_contains:
        lowered = reason.lower()
        if not any(token.lower() in lowered for token in reason_contains):
            return False
    return True


def _baseline_rows(rows: list[dict[str, str]], baseline_runs: list[str]) -> list[dict[str, str]]:
    out = []
    for run in baseline_runs:
        matched = next((row for row in rows if str(row.get("run", "")).strip() == str(run).strip()), None)
        if matched is None:
            raise KeyError(f"Baseline run not found: {run}")
        out.append(matched)
    return out


def _safe_tag(text: str) -> str:
    raw = str(text).strip().replace(" ", "_")
    cleaned = "".join(ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in raw)
    while "__" in cleaned:
        cleaned = cleaned.replace("__", "_")
    return cleaned.strip("_") or "item"


def main() -> int:
    parser = argparse.ArgumentParser(description="Build one or more round-1 external-baseline VLM manifests directly from a family bestfew handoff table.")
    parser.add_argument("--handoff-csv", type=Path, required=True)
    parser.add_argument("--baseline-manifest", type=Path, required=True)
    parser.add_argument("--baseline-runs", nargs="+", default=["Seedream_repaired750", "SaMAM_2250"])
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--family-label-prefix", required=True, help="Prefix for candidate run labels, for example AttnGWOT")
    parser.add_argument("--family-method", default="LBM")
    parser.add_argument("--epochs", nargs="*", default=[])
    parser.add_argument("--reason-contains", nargs="*", default=[])
    parser.add_argument("--index-json", type=Path, default=None)
    args = parser.parse_args()

    handoff_csv = Path(args.handoff_csv).resolve()
    baseline_manifest = Path(args.baseline_manifest).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    index_json = Path(args.index_json).resolve() if args.index_json is not None else output_dir / "manifest_index.json"

    baseline_rows = _baseline_rows(_read_rows(baseline_manifest), list(args.baseline_runs))
    handoff_rows = _read_rows(handoff_csv)
    epochs = {str(item).strip() for item in args.epochs if str(item).strip()}
    selected = [row for row in handoff_rows if _match_row(row, epochs=epochs, reason_contains=list(args.reason_contains))]
    if not selected:
        raise RuntimeError("No handoff rows matched the requested filters.")

    outputs: list[dict[str, str]] = []
    for row in selected:
        epoch = str(row.get("epoch", "")).strip()
        candidate_label = f"{args.family_label_prefix}_{epoch.replace('epoch_', 'e')}"
        reason_tag = _safe_tag(str(row.get("reason", "")).strip())
        out_csv = output_dir / f"{_safe_tag(args.family_label_prefix)}_{epoch}_{reason_tag}.csv"
        manifest_rows = [
            {
                "method": str(base.get("method", "")).strip(),
                "run": str(base.get("run", "")).strip(),
                "images_dir": str(base.get("images_dir", "")).strip(),
                "source_root": str(base.get("source_root", "")).strip(),
                "metrics_csv": str(base.get("metrics_csv", "")).strip(),
            }
            for base in baseline_rows
        ]
        manifest_rows.append(
            {
                "method": str(args.family_method),
                "run": candidate_label,
                "images_dir": str(row.get("images_dir", "")).strip(),
                "source_root": str(row.get("source_root", "")).strip() or str((baseline_rows[0].get("source_root", "")).strip()),
                "metrics_csv": str(row.get("metrics_csv", "")).strip(),
            }
        )
        with out_csv.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["method", "run", "images_dir", "source_root", "metrics_csv"])
            writer.writeheader()
            writer.writerows(manifest_rows)
        outputs.append(
            {
                "epoch": epoch,
                "reason": str(row.get("reason", "")).strip(),
                "candidate_label": candidate_label,
                "manifest_csv": str(out_csv),
            }
        )

    index_payload = {
        "handoff_csv": str(handoff_csv),
        "baseline_manifest": str(baseline_manifest),
        "baseline_runs": list(args.baseline_runs),
        "family_label_prefix": str(args.family_label_prefix),
        "family_method": str(args.family_method),
        "outputs": outputs,
    }
    index_json.write_text(json.dumps(index_payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(index_json)
    for item in outputs:
        print(item["manifest_csv"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
