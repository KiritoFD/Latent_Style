"""
Baseline Reproduction Pipeline - Main Orchestrator
For AAAI 2027 Paper Experiments

Moved to Related_Works/ for standalone operation.
All paths are relative to Related_Works/.
"""
import os
import sys
import subprocess
import argparse
from pathlib import Path

PIPELINE_ROOT = Path(__file__).parent.resolve()

ZERO_SHOT = ["styleid", "cyclegan_turbo"]
TRAINING = ["cut", "s2wat", "style_aligned", "blora"]
ALL_BASELINES = ZERO_SHOT + TRAINING
ALL_STYLES = ["monet", "vangogh", "ukiyoe", "cezanne", "Hayao"]


def run_baseline(baseline, style, skip_gen=False, skip_eval=False, smoke=False):
    print(f"\n{'='*60}")
    print(f"  Baseline: {baseline}  |  Style: {style}")
    print(f"{'='*60}")

    if not skip_gen:
        script = PIPELINE_ROOT / "scripts" / f"run_{baseline}.py"
        if baseline == "cut":
            script = PIPELINE_ROOT / "scripts" / "copy_cut_results.py"
            subprocess.run([sys.executable, str(script)], check=True)
        elif script.exists():
            cmd = [sys.executable, str(script), "--style", style]
            if smoke and baseline in TRAINING:
                cmd += ["--mode", "smoke"]
            elif smoke and baseline in ZERO_SHOT:
                cmd += ["--max_images", "5"]
            subprocess.run(cmd, check=True)
        else:
            print(f"[WARN] Script not found: {script}")

    if not skip_eval:
        eval_script = PIPELINE_ROOT / "evaluation" / "eval_all_baselines.py"
        subprocess.run([sys.executable, str(eval_script),
                       "--baseline", baseline, "--style", style], check=True)


def main():
    parser = argparse.ArgumentParser(description="Baseline Pipeline")
    parser.add_argument("--baselines", nargs="+", default=["s2wat"],
                       help=f"Baselines: {ALL_BASELINES}")
    parser.add_argument("--styles", nargs="+", default=["monet"],
                       help=f"Styles: {ALL_STYLES}")
    parser.add_argument("--smoke", action="store_true", help="Smoke test mode")
    parser.add_argument("--skip-generation", action="store_true")
    parser.add_argument("--skip-evaluation", action="store_true")
    args = parser.parse_args()

    for b in args.baselines:
        if b not in ALL_BASELINES:
            print(f"Unknown baseline: {b}")
            return

    for b in args.baselines:
        for s in args.styles:
            (PIPELINE_ROOT / "results" / b / s).mkdir(parents=True, exist_ok=True)
            try:
                run_baseline(b, s, args.skip_generation, args.skip_evaluation, args.smoke)
            except Exception as e:
                print(f"[ERROR] {b}/{s}: {e}")

    print("\n=== Pipeline Complete ===")


if __name__ == "__main__":
    main()
