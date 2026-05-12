"""
Copy existing CUT inference results from Related_Works/runs to baseline_pipeline/results.
CUT is already fully trained and evaluated - no retraining needed.
Supports 5x5 format: all content styles -> all target styles.
"""
import os
import shutil
import argparse
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent.resolve()
PIPELINE_ROOT = SCRIPT_DIR.parent
WORKSPACE_ROOT = PIPELINE_ROOT.parent.parent
CUT_RESULTS = WORKSPACE_ROOT / "Related_Works" / "runs" / "cut_5x5"
OUTPUT_DIR = PIPELINE_ROOT / "results" / "cut"

ALL_STYLES = ["photo", "monet", "vangogh", "ukiyoe", "cezanne", "Hayao"]


def copy_cut_results(output_root: Path = OUTPUT_DIR, source_dir: Path | None = None):
    source_dir = source_dir or (CUT_RESULTS / "infer_5x5" / "images")
    if not source_dir.exists():
        print(f"[ERROR] CUT results not found at {source_dir}")
        return 1

    for style in ALL_STYLES:
        (output_root / style).mkdir(parents=True, exist_ok=True)

    files = list(source_dir.glob("*.jpg")) + list(source_dir.glob("*.png"))
    count = 0

    for f in files:
        # Format: {content}_{id}_to_{target}.jpg or {content}_{id}_flip_to_{target}.jpg
        name = f.stem
        if "_to_" not in name:
            continue
        tgt_style = name.split("_to_")[-1]
        if tgt_style not in ALL_STYLES:
            continue

        dst = output_root / tgt_style / f.name
        if not dst.exists():
            shutil.copy2(f, dst)
        count += 1

    print(f"[CUT] Copied {count} results")
    for style in ALL_STYLES:
        n = len(list((output_root / style).glob("*")))
        if n > 0:
            print(f"  {style}: {n} images")

    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_root", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--source_dir", type=Path, default=CUT_RESULTS / "infer_5x5" / "images")
    args = parser.parse_args()
    raise SystemExit(copy_cut_results(args.output_root.resolve(), args.source_dir.resolve()))
