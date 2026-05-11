"""
Unified Launch Script for Baseline Reproduction Pipeline
For AAAI 2027 Paper Experiments

Baselines (all use SD1.5 VAE):
    - s2wat:        Training required (wavelet transformer + VGG)
    - styleid:      Zero-shot (diffusers img2img)
    - style_aligned: Zero-shot (ControlNet Canny + SD1.5)
    - cut:          Already done (copy existing results)

Usage:
    # Smoke test (1 epoch, 5 images) for training baselines:
    python launch_all.py --smoke

    # Full training for all baselines:
    python launch_all.py --full

    # Train specific baselines:
    python launch_all.py --baselines s2wat style_aligned --styles monet vangogh

    # Zero-shot inference only (no training):
    python launch_all.py --zero-shot --styles monet vangogh ukiyoe cezanne

    # Copy CUT results (already done):
    python launch_all.py --baselines cut
"""
import os
import sys
import subprocess
import argparse
import time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

# Ensure Python 3.12+ is used
if sys.version_info < (3, 12):
    # Try to re-exec with Python 3.12
    py312 = Path(r"C:\Users\xy\AppData\Local\Programs\Python\Python312\python.exe")
    if py312.exists():
        print(f"[RE-EXEC] Python {sys.version} < 3.12, switching to {py312}")
        os.execv(str(py312), [str(py312)] + sys.argv)
    else:
        print(f"[WARN] Python {sys.version} detected. Recommend Python 3.12+")

PIPELINE_ROOT = Path(__file__).parent.resolve()
REPO_ROOT = PIPELINE_ROOT.parent.parent  # g:\GitHub\Latent_Style
STYLE_DATA = REPO_ROOT / "style_data"
OVERFIT50 = STYLE_DATA / "overfit50"
TRAIN_DATA = STYLE_DATA / "train"

# Available baselines
# styleid: zero-shot (diffusers img2img approach, SD1.5)
# style_aligned: zero-shot (ControlNet Canny + SD1.5)
# cut: already done (copy existing results)
# s2wat: training required (wavelet transformer + VGG)
# samst: training required (TransformerNet + VGG16, lightweight)
ZERO_SHOT = ["styleid", "style_aligned"]
TRAINING_REQUIRED = ["cut", "s2wat", "samst"]
ALL_BASELINES = ZERO_SHOT + TRAINING_REQUIRED
ALL_STYLES = ["monet", "vangogh", "ukiyoe", "cezanne", "Hayao"]


def run_cmd(cmd, cwd=None, desc=""):
    """Run command with live output"""
    print(f"\n{'='*60}")
    print(f"[LAUNCH] {desc}")
    print(f"  CMD: {' '.join(str(c) for c in cmd)}")
    print(f"  CWD: {cwd or Path.cwd()}")
    print(f"{'='*60}")
    t0 = time.time()
    result = subprocess.run([str(c) for c in cmd], cwd=str(cwd) if cwd else None)
    elapsed = time.time() - t0
    status = "OK" if result.returncode == 0 else f"FAIL(rc={result.returncode})"
    print(f"[{status}] {desc} took {elapsed:.1f}s")
    return result.returncode


def launch_s2wat(style, smoke=False):
    """Train + infer S2WAT for one style"""
    script = PIPELINE_ROOT / "scripts" / "run_s2wat.py"
    mode = "smoke" if smoke else "all"
    return run_cmd(
        [sys.executable, script, "--style", style, "--mode", mode],
        cwd=PIPELINE_ROOT,
        desc=f"S2WAT train+infer [{style}] ({mode})"
    )


def launch_style_aligned(style, smoke=False):
    """Infer StyleAligned for one style (zero-shot, no training)"""
    script = PIPELINE_ROOT / "scripts" / "run_style_aligned.py"
    n = 5 if smoke else 0  # 0 = all
    return run_cmd(
        [sys.executable, script, "--style", style, "--max_images", str(n)],
        cwd=PIPELINE_ROOT,
        desc=f"StyleAligned infer [{style}]"
    )


def launch_styleid(style, smoke=False):
    """Infer StyleID for one style (zero-shot)"""
    script = PIPELINE_ROOT / "scripts" / "run_styleid.py"
    n = 5 if smoke else 0
    return run_cmd(
        [sys.executable, script, "--style", style, "--max_images", str(n)],
        cwd=PIPELINE_ROOT,
        desc=f"StyleID infer [{style}]"
    )


def launch_samst(style, smoke=False):
    """Train + infer SaMST for one style"""
    script = PIPELINE_ROOT / "scripts" / "run_samst.py"
    mode = "smoke" if smoke else "all"
    return run_cmd(
        [sys.executable, script, "--style", style, "--mode", mode],
        cwd=PIPELINE_ROOT,
        desc=f"SaMST train+infer [{style}] ({mode})"
    )


def launch_cut(style):
    """Copy existing CUT results"""
    script = PIPELINE_ROOT / "scripts" / "copy_cut_results.py"
    return run_cmd(
        [sys.executable, script],
        cwd=PIPELINE_ROOT,
        desc=f"CUT copy results [{style}]"
    )


def launch_eval(baseline, style):
    """Run evaluation for one baseline+style"""
    script = PIPELINE_ROOT / "evaluation" / "eval_all_baselines.py"
    return run_cmd(
        [sys.executable, script, "--baseline", baseline, "--style", style],
        cwd=PIPELINE_ROOT,
        desc=f"Eval {baseline}/{style}"
    )


LAUNCHERS = {
    "s2wat": launch_s2wat,
    "samst": launch_samst,
    "style_aligned": launch_style_aligned,
    "styleid": launch_styleid,
    "cut": launch_cut,
}

DEFAULT_BASELINES = ["s2wat", "samst", "style_aligned", "styleid"]


def main():
    parser = argparse.ArgumentParser(description="Baseline Reproduction Launch Script")
    parser.add_argument("--baselines", nargs="+", default=None,
                       help=f"Baselines to run. Default: all. Choices: {ALL_BASELINES}")
    parser.add_argument("--styles", nargs="+", default=ALL_STYLES,
                       help=f"Styles to process. Default: all. Choices: {ALL_STYLES}")
    parser.add_argument("--smoke", action="store_true",
                       help="Smoke test mode: 1 epoch training, 5 images inference")
    parser.add_argument("--full", action="store_true",
                       help="Full training mode")
    parser.add_argument("--zero-shot", action="store_true",
                       help="Only run zero-shot baselines (styleid, style_aligned)")
    parser.add_argument("--eval-only", action="store_true",
                       help="Only run evaluation, skip generation/training")
    parser.add_argument("--parallel", type=int, default=1,
                       help="Number of parallel workers (default: 1, sequential)")
    parser.add_argument("--skip-eval", action="store_true",
                       help="Skip evaluation after generation")
    args = parser.parse_args()

    # Determine baselines
    if args.baselines is None:
        if args.zero_shot:
            baselines = ZERO_SHOT
        else:
            baselines = DEFAULT_BASELINES
    else:
        baselines = args.baselines

    # Validate
    for b in baselines:
        if b not in ALL_BASELINES:
            print(f"Error: Unknown baseline '{b}'. Available: {ALL_BASELINES}")
            return

    smoke = args.smoke and not args.full

    print(f"\n{'='*60}")
    print(f"BASELINE PIPELINE LAUNCH")
    print(f"  Baselines: {baselines}")
    print(f"  Styles: {args.styles}")
    print(f"  Mode: {'SMOKE (1 epoch, 5 imgs)' if smoke else 'FULL'}")
    print(f"  Parallel: {args.parallel}")
    print(f"{'='*60}\n")

    # Create results dirs
    for b in baselines:
        for s in args.styles:
            (PIPELINE_ROOT / "results" / b / s).mkdir(parents=True, exist_ok=True)

    # Collect tasks
    tasks = []
    for baseline in baselines:
        if args.eval_only:
            for style in args.styles:
                tasks.append(("eval", baseline, style))
        else:
            for style in args.styles:
                tasks.append(("run", baseline, style))
            if not args.skip_eval:
                for style in args.styles:
                    tasks.append(("eval", baseline, style))

    # Execute
    if args.parallel <= 1:
        for kind, baseline, style in tasks:
            if kind == "run":
                launcher = LAUNCHERS[baseline]
                if baseline == "cut":
                    launcher(style)
                else:
                    launcher(style, smoke=smoke)
            else:
                launch_eval(baseline, style)
    else:
        # Parallel execution (limited - GPU-bound tasks should not fully parallelize)
        with ProcessPoolExecutor(max_workers=args.parallel) as executor:
            futures = {}
            for kind, baseline, style in tasks:
                if kind == "run":
                    launcher = LAUNCHERS[baseline]
                    if baseline == "cut":
                        f = executor.submit(launcher, style)
                    else:
                        f = executor.submit(launcher, style, smoke)
                else:
                    f = executor.submit(launch_eval, baseline, style)
                futures[f] = f"{kind}/{baseline}/{style}"

            for f in as_completed(futures):
                name = futures[f]
                rc = f.result()
                print(f"[{'OK' if rc == 0 else 'FAIL'}] {name}")

    print(f"\n{'='*60}")
    print("PIPELINE COMPLETE")
    print(f"Results: {PIPELINE_ROOT / 'results'}")
    print(f"Metrics: {PIPELINE_ROOT / 'results' / 'metrics.csv'}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
