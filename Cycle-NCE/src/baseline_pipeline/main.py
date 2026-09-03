"""
Baseline Reproduction Pipeline Main Script
For AAAI 2027 Paper Experiments
"""
import os
import sys
import subprocess
import json
from pathlib import Path
import argparse

# Configuration
AVAILABLE_BASELINES = {
    "zero_shot": ["styleid", "cyclegan_turbo"],
    "training_required": ["cut", "s2wat", "style_aligned", "blora"]
}

ALL_BASELINES = AVAILABLE_BASELINES["zero_shot"] + AVAILABLE_BASELINES["training_required"]

STYLES = ["photo", "monet", "vangogh", "ukiyoe", "cezanne"]  # 5 styles total for 5x5 conversion, all equal (can be content/style)

def run_command(cmd, cwd=None):
    """Run a command and return output"""
    print(f"\n=== Running: {' '.join(cmd)} ===")
    result = subprocess.run(cmd, cwd=cwd, check=True)
    return result

def run_baseline(baseline_name, style, skip_generation=False, skip_evaluation=False):
    """Run a single baseline for a single style"""
    print(f"\n{'='*60}")
    print(f"Processing baseline: {baseline_name}, style: {style}")
    print(f"{'='*60}")
    
    if not skip_generation:
        if baseline_name == "cut":
            # Copy existing CUT results instead of retraining
            print("Using existing CUT results (no retraining)...")
            cmd = [sys.executable, "./scripts/copy_cut_results.py"]
            run_command(cmd)
        else:
            # Run generation/training script
            script_path = f"./scripts/run_{baseline_name}.py"
            if not os.path.exists(script_path):
                print(f"Warning: Script for {baseline_name} not implemented yet, skipping generation")
            else:
                cmd = [sys.executable, script_path, "--style", style]
                run_command(cmd)
    
    if not skip_evaluation:
        # Run evaluation
        eval_script = "./evaluation/eval_all_baselines.py"
        cmd = [sys.executable, eval_script, "--baseline", baseline_name, "--style", style]
        run_command(cmd, cwd="evaluation")

def main():
    parser = argparse.ArgumentParser(description="Baseline Reproduction Pipeline")
    parser.add_argument("--baselines", nargs="+", default=["styleid", "cyclegan_turbo"], 
                       help=f"Baselines to run: {ALL_BASELINES}")
    parser.add_argument("--styles", nargs="+", default=["monet"], 
                       help=f"Styles to run: {STYLES}")
    parser.add_argument("--skip-generation", action="store_true", help="Skip generation phase, only run evaluation")
    parser.add_argument("--skip-evaluation", action="store_true", help="Skip evaluation phase, only run generation")
    parser.add_argument("--list-baselines", action="store_true", help="List all available baselines")
    parser.add_argument("--list-styles", action="store_true", help="List all available styles")
    
    args = parser.parse_args()
    
    if args.list_baselines:
        print("Available baselines:")
        print("Zero-shot (no training required):")
        for b in AVAILABLE_BASELINES["zero_shot"]:
            print(f"  - {b}")
        print("\nTraining required:")
        for b in AVAILABLE_BASELINES["training_required"]:
            print(f"  - {b}")
        return
    
    if args.list_styles:
        print("Available styles:")
        for s in STYLES:
            print(f"  - {s}")
        return
    
    # Validate inputs
    for b in args.baselines:
        if b not in ALL_BASELINES:
            print(f"Error: Unknown baseline {b}")
            print(f"Available baselines: {ALL_BASELINES}")
            return
    
    # Create results directories
    for b in args.baselines:
        for s in args.styles:
            os.makedirs(f"./results/{b}/{s}", exist_ok=True)
    
    # Run all baselines and styles
    print(f"Starting baseline pipeline with baselines: {args.baselines}, styles: {args.styles}")
    
    for baseline in args.baselines:
        for style in args.styles:
            try:
                run_baseline(baseline, style, args.skip_generation, args.skip_evaluation)
            except Exception as e:
                print(f"Error processing {baseline} - {style}: {e}")
                print("Continuing with next task...")
    
    print("\n=== Pipeline completed ===")
    print("Results saved to ./results/")
    print("Metrics saved to ./results/metrics.csv")

if __name__ == "__main__":
    main()
