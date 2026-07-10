"""Extract four key metrics (CLIP-S, LPIPS, DINO-S, DINO-C) from an eval directory."""
import sys
import json
import csv
from pathlib import Path


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: _710_extract_run.py <eval_dir> [run_name]")
        sys.exit(1)
    eval_dir = Path(sys.argv[1])
    run_name = sys.argv[2] if len(sys.argv) > 2 else eval_dir.name

    clip_s = lpips = dino_s = dino_c = 0.0

    # Prefer dino_summary.json (has all four metrics aggregated)
    dino_summary_path = eval_dir / "dino_summary.json"
    if dino_summary_path.exists():
        summary = json.loads(dino_summary_path.read_text(encoding="utf-8"))
        clip_s = float(summary.get("all_clip_s", 0.0))
        lpips = float(summary.get("all_lpips", 0.0))
        dino_s = float(summary.get("all_dino_s", 0.0))
        dino_c = float(summary.get("all_dino_c", 0.0))
    else:
        # Fallback: read metrics.csv and dino_metrics.csv separately
        metrics_csv = eval_dir / "metrics.csv"
        if metrics_csv.exists():
            rows = list(csv.DictReader(metrics_csv.open(encoding="utf-8-sig")))
            if rows:
                clip_s = sum(float(r.get("clip_style", 0)) for r in rows) / len(rows)
                lpips = sum(float(r.get("content_lpips", 0)) for r in rows) / len(rows)
        dino_csv = eval_dir / "dino_metrics.csv"
        if dino_csv.exists():
            rows = list(csv.DictReader(dino_csv.open(encoding="utf-8-sig")))
            if rows:
                dino_s = sum(float(r.get("dino_style", 0)) for r in rows) / len(rows)
                dino_c = sum(float(r.get("dino_content", 0)) for r in rows) / len(rows)

    result = {
        "run_name": run_name,
        "clip_s": round(clip_s, 4),
        "lpips": round(lpips, 4),
        "dino_s": round(dino_s, 4),
        "dino_c": round(dino_c, 4),
    }
    output_path = eval_dir / "extracted_metrics.json"
    output_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
