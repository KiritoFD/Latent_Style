"""
Prepare baseline results for SB evaluation.
Creates unified images/ directory per baseline with proper naming.
"""
import shutil
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent.resolve()
PIPELINE_ROOT = SCRIPT_DIR.parent
RESULTS_DIR = PIPELINE_ROOT / "results"

BASELINE_STYLES = {
    "s2wat": ["monet", "vangogh", "cezanne", "Hayao"],
    "samst": ["monet", "vangogh", "cezanne"],
    "styleid": ["monet", "vangogh", "cezanne", "Hayao"],
    "cut": ["monet", "vangogh", "cezanne", "Hayao"],
}


def prepare_baseline(baseline):
    """Copy/symlink all style images into a unified images/ dir."""
    bl_dir = RESULTS_DIR / baseline
    images_dir = bl_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    count = 0
    for style in BASELINE_STYLES.get(baseline, []):
        # Images may be in results/{bl}/{style}/ or results/{bl}/{style}/images/
        style_dir = RESULTS_DIR / baseline / style
        candidates = list(style_dir.glob("*.jpg")) + list(style_dir.glob("*.png"))
        if not candidates:
            sub = style_dir / "images"
            if sub.exists():
                candidates = list(sub.glob("*.jpg")) + list(sub.glob("*.png"))

        for img in candidates:
            dst = images_dir / img.name
            if not dst.exists():
                shutil.copy2(str(img), str(dst))
            count += 1

    print(f"[{baseline}] {count} images -> {images_dir}")
    return count


def main():
    for bl in BASELINE_STYLES:
        prepare_baseline(bl)


if __name__ == "__main__":
    main()
