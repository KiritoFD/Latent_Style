"""
Evaluate baseline results using the SchrodingerBridge evaluation script.
Calls utils.run_evaluation with --reuse_generated to compute metrics
on existing generated images (LPIPS-VGG, CLIP_Style, CLIP_Content).

Usage:
    python eval_with_sb.py --baseline s2wat --style monet
    python eval_with_sb.py --baseline all --style all
"""
import os
import sys
import subprocess
import argparse
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent.resolve()
PIPELINE_ROOT = SCRIPT_DIR.parent
REPO_ROOT = PIPELINE_ROOT.parent.parent
SB_ROOT = REPO_ROOT / "SchrodingerBridge"
SB_SRC = SB_ROOT / "src"
OVERFIT50 = REPO_ROOT / "style_data" / "overfit50"
RESULTS_DIR = PIPELINE_ROOT / "results"

ALL_STYLES = ["monet", "vangogh", "ukiyoe", "cezanne", "Hayao"]
BASELINE_STYLES = {
    "s2wat": ["monet", "vangogh", "cezanne", "Hayao"],
    "samst": ["monet", "vangogh", "cezanne", "ukiyoe"],
    "styleid": ["monet", "vangogh", "cezanne", "Hayao"],
    "cut": ["monet", "vangogh", "cezanne", "Hayao"],
}


def eval_baseline_style(baseline: str, style: str) -> int:
    """Run SB evaluation on one baseline+style directory."""
    result_dir = RESULTS_DIR / baseline / style
    if not result_dir.exists() or not any(result_dir.glob("*.jpg")):
        print(f"[SKIP] {baseline}/{style} - no images")
        return 0

    count = len(list(result_dir.glob("*.jpg")))
    print(f"\n[SB-EVAL] {baseline}/{style} ({count} images)")

    output_dir = result_dir / "sb_eval"

    cmd = [
        sys.executable,
        str(SB_ROOT / "run_evaluation.py"),
        "--output", str(output_dir),
        "--test_dir", str(OVERFIT50),
        "--style_subdirs", style,
        "--reuse_generated",
        "--force_regen",
        "--eval_only_lpips_clip_style",
        "--no-eval_enable_art_fid",
        "--no-eval_enable_kid",
    ]

    # Set working directory to SB root so relative paths resolve
    env = os.environ.copy()
    env["PYTHONPATH"] = str(SB_SRC) + os.pathsep + env.get("PYTHONPATH", "")

    result = subprocess.run(cmd, cwd=str(SB_ROOT), env=env)
    if result.returncode == 0:
        print(f"  [OK] Results in {output_dir}")
    else:
        print(f"  [FAIL] Exit code {result.returncode}")
    return result.returncode


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=str, required=True,
                       help="Baseline name or 'all'")
    parser.add_argument("--style", type=str, required=True,
                       help="Style name or 'all'")
    args = parser.parse_args()

    baselines = list(BASELINE_STYLES.keys()) if args.baseline == "all" else [args.baseline]
    styles = ALL_STYLES if args.style == "all" else [args.style]

    failures = 0
    for bl in baselines:
        if bl not in BASELINE_STYLES:
            print(f"[ERROR] Unknown baseline: {bl}")
            failures += 1
            continue
        valid_styles = styles if args.style == "all" else [s for s in styles if s in BASELINE_STYLES[bl]]
        for s in valid_styles:
            rc = eval_baseline_style(bl, s)
            if rc != 0:
                failures += 1

    print(f"\n{'='*50}")
    print(f"Done. Failures: {failures}")
    return 1 if failures > 0 else 0


if __name__ == "__main__":
    sys.exit(main())
