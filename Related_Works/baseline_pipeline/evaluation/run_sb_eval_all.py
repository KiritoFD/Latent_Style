"""
Batch run SchrodingerBridge evaluation on all baseline results.
Collects metrics from summary.json into a unified CSV.
"""
import json
import subprocess
import sys
import csv
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent.resolve()
PIPELINE_ROOT = SCRIPT_DIR.parent
REPO_ROOT = PIPELINE_ROOT.parent.parent
SB_ROOT = REPO_ROOT / "SchrodingerBridge"
SB_SRC = SB_ROOT / "src"
OVERFIT50 = REPO_ROOT / "style_data" / "overfit50"
RESULTS_DIR = PIPELINE_ROOT / "results"
CLIP_PATH = REPO_ROOT / "Cycle-NCE" / "eval_cache" / "manual_clip" / "openai-clip-vit-base-patch32"

BASELINE_STYLES = {
    "s2wat": ["monet", "vangogh", "cezanne", "Hayao"],
    "samst": ["monet", "vangogh", "cezanne"],  # ukiyoe skipped: no overfit50/ukiyoe
    "styleid": ["monet", "vangogh", "cezanne", "Hayao"],
    "cut": ["monet", "vangogh", "cezanne", "Hayao"],
}


def run_eval(baseline, style):
    """Run SB evaluation for one baseline+style."""
    result_dir = RESULTS_DIR / baseline / style
    if not result_dir.exists() or not any(result_dir.glob("*.jpg")):
        print(f"[SKIP] {baseline}/{style}")
        return None

    summary_path = result_dir / "summary.json"
    # Check if summary already exists and has valid metrics
    if summary_path.exists():
        try:
            data = json.loads(summary_path.read_text(encoding="utf-8"))
            a = data.get("analysis", {}).get("all_pairs_overview", {})
            if a.get("clip_style") and a.get("clip_style") > 0:
                print(f"[CACHED] {baseline}/{style}")
                return summary_path
        except Exception:
            pass

    count = len(list(result_dir.glob("*.jpg")))
    print(f"\n[SB-EVAL] {baseline}/{style} ({count} imgs)")

    cmd = [
        sys.executable, "-m", "utils.run_evaluation",
        "--output", str(result_dir),
        "--test_dir", str(OVERFIT50),
        "--style_subdirs", "photo,monet,vangogh,cezanne,Hayao",
        "--reuse_generated", "--force_regen",
        "--no-eval_enable_art_fid", "--no-eval_enable_kid",
        "--clip_model_name", str(CLIP_PATH),
    ]

    result = subprocess.run(cmd, cwd=str(SB_SRC), capture_output=False)
    if result.returncode != 0:
        print(f"  [FAIL] exit {result.returncode}")
        return None
    return summary_path


def collect_metrics():
    """Collect metrics from all summary.json files."""
    rows = []
    for baseline, styles in BASELINE_STYLES.items():
        for style in styles:
            summary_path = RESULTS_DIR / baseline / style / "summary.json"
            if not summary_path.exists():
                continue
            data = json.loads(summary_path.read_text(encoding="utf-8"))
            a = data.get("analysis", {}).get("all_pairs_overview", {})
            rows.append({
                "experiment_id": f"{baseline}/{style}",
                "clip_style": a.get("clip_style"),
                "clip_content": a.get("clip_content"),
                "content_lpips": a.get("content_lpips"),
            })
    return rows


def main():
    # Run evaluations
    for baseline, styles in BASELINE_STYLES.items():
        for style in styles:
            run_eval(baseline, style)

    # Collect and save
    rows = collect_metrics()
    out_csv = RESULTS_DIR / "metrics_sb.csv"
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["experiment_id", "clip_style", "clip_content", "content_lpips"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"\n{'='*60}")
    print(f"Saved {len(rows)} rows to {out_csv}")
    for r in rows:
        cs = f"{r['clip_style']:.4f}" if r['clip_style'] else "N/A"
        cc = f"{r['clip_content']:.4f}" if r['clip_content'] else "N/A"
        lp = f"{r['content_lpips']:.4f}" if r['content_lpips'] else "N/A"
        print(f"  {r['experiment_id']:25s}  clip_style={cs}  clip_content={cc}  lpips={lp}")


if __name__ == "__main__":
    main()
