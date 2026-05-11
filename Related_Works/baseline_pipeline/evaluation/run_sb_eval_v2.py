"""
Run SB evaluation on baselines using unified images/ directories.
One eval call per baseline (all target styles together).
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

BASELINES = ["s2wat", "samst", "styleid", "cut"]


def run_eval(baseline):
    bl_dir = RESULTS_DIR / baseline
    images_dir = bl_dir / "images"
    if not images_dir.exists() or not any(images_dir.glob("*.jpg")):
        print(f"[SKIP] {baseline} - no images/")
        return None

    summary_path = bl_dir / "summary.json"
    if summary_path.exists():
        try:
            data = json.loads(summary_path.read_text(encoding="utf-8"))
            a = data.get("analysis", {}).get("all_pairs_overview", {})
            if a.get("clip_style") and a.get("clip_style") > 0:
                print(f"[CACHED] {baseline}")
                return summary_path
        except Exception:
            pass

    count = len(list(images_dir.glob("*.jpg")))
    print(f"\n[SB-EVAL] {baseline} ({count} imgs)")

    cmd = [
        sys.executable, "-m", "utils.run_evaluation",
        "--output", str(bl_dir),
        "--test_dir", str(OVERFIT50),
        "--style_subdirs", "photo,monet,vangogh,cezanne,Hayao",
        "--reuse_generated", "--force_regen",
        "--no-eval_enable_art_fid", "--no-eval_enable_kid",
        "--clip_model_name", str(CLIP_PATH),
    ]

    result = subprocess.run(cmd, cwd=str(SB_SRC))
    if result.returncode != 0:
        print(f"  [FAIL] exit {result.returncode}")
        return None
    return summary_path


def collect_metrics():
    rows = []
    for bl in BASELINES:
        summary_path = RESULTS_DIR / bl / "summary.json"
        if not summary_path.exists():
            continue
        data = json.loads(summary_path.read_text(encoding="utf-8"))
        a = data.get("analysis", {}).get("all_pairs_overview", {})
        rows.append({
            "experiment_id": bl,
            "clip_style": a.get("clip_style"),
            "clip_content": a.get("clip_content"),
            "content_lpips": a.get("content_lpips"),
        })
    return rows


def main():
    for bl in BASELINES:
        run_eval(bl)

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
        print(f"  {r['experiment_id']:12s}  clip_style={cs}  clip_content={cc}  lpips={lp}")


if __name__ == "__main__":
    main()
