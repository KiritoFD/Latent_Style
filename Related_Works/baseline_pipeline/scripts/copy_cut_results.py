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
REPO_ROOT = PIPELINE_ROOT.parent.parent
CUT_RESULTS = REPO_ROOT / "Related_Works" / "runs" / "cut_5x5"
OUTPUT_DIR = PIPELINE_ROOT / "results" / "cut"

ALL_STYLES = ["photo", "monet", "vangogh", "ukiyoe", "cezanne", "Hayao"]


def copy_cut_results():
    source_dir = CUT_RESULTS / "infer_val_clean_5x5" / "images"
    if not source_dir.exists():
        print(f"[ERROR] CUT results not found at {source_dir}")
        return 1

    for style in ALL_STYLES:
        (OUTPUT_DIR / style).mkdir(parents=True, exist_ok=True)

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

        dst = OUTPUT_DIR / tgt_style / f.name
        if not dst.exists():
            shutil.copy2(f, dst)
        count += 1

    print(f"[CUT] Copied {count} results")
    for style in ALL_STYLES:
        n = len(list((OUTPUT_DIR / style).glob("*")))
        if n > 0:
            print(f"  {style}: {n} images")

    return 0


if __name__ == "__main__":
    copy_cut_results()
