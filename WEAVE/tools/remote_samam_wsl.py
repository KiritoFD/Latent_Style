"""SaMam training + inference in WSL environment.
Runs on the same GPU but uses WSL mamba-ssm environment.
Generates 750 outputs in standard naming format.
Records training and inference time.
"""
import os
import sys
import time
import shutil
import subprocess
from pathlib import Path

# Force UTF-8 stdout/stderr
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
sys.stderr.reconfigure(encoding='utf-8', errors='replace')

# ── Configuration ──
STYLES = ['Early_Renaissance', 'Impressionism', 'Minimalism', 'Rococo', 'Ukiyo_e']
TEST_DIR = Path('/mnt/i/wikiart_distinct5_samam_512_classview/test')
TRAIN_DIR = Path('/mnt/i/wikiart_distinct5_samam_512_classview/train')
SAMAM_REPO = Path('/mnt/i/GitHub/Latent_Style/Related_Works/repos/SaMam')
OUT_ROOT = Path('/mnt/i/GitHub/Latent_Style/SchrodingerBridge/exp/baseline_v2')
IMAGE_ROOT = OUT_ROOT / 'images'
SAMAM_OUT = IMAGE_ROOT / 'samam'
CKPT_DIR = OUT_ROOT / 'checkpoints' / 'samam'
DATA_DIR = OUT_ROOT / 'data' / 'samam_data'

# Training config (reduced from 200k iterations for time budget)
# 5000 iters at batch_size=4, image_size=256 ~ 30-45 min on 3060
TRAIN_ITERATIONS = 5000
BATCH_SIZE = 4  # SaMam default is 8; reduce to 4 for 12GB GPU
IMAGE_SIZE = 256  # SaMam default; 512 would OOM
VAL_INTERVAL = 500

PYTHON = '/home/xy/venvs/samam312/bin/python'


def is_done():
    return (SAMAM_OUT / '_DONE').exists()


def mark_done(count):
    (SAMAM_OUT / '_DONE').write_text(f'{count} images\n{time.strftime("%Y-%m-%d %H:%M:%S")}')


def prepare_data():
    """Prepare content/style directories for SaMam training."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    content_dir = DATA_DIR / 'content'
    style_dir = DATA_DIR / 'style'
    content_dir.mkdir(exist_ok=True)
    style_dir.mkdir(exist_ok=True)

    # Use train images as both content and style sources
    # Subsample to ~200/style for faster training (1000 total)
    MAX_PER_STYLE = 200
    for style in STYLES:
        src_dir = TRAIN_DIR / style
        count = 0
        for f in sorted(src_dir.iterdir()):
            if f.suffix.lower() not in ('.jpg', '.png', '.jpeg'):
                continue
            if count >= MAX_PER_STYLE:
                break
            # Content: copy as-is with style prefix
            dst_c = content_dir / f'{style}__{f.name}'
            if not dst_c.exists():
                shutil.copy2(str(f), str(dst_c))
            # Style: same images can serve as style references
            dst_s = style_dir / f'{style}__{f.name}'
            if not dst_s.exists():
                shutil.copy2(str(f), str(dst_s))
            count += 1

    print(f"  Prepared {len(list(content_dir.iterdir()))} content, "
          f"{len(list(style_dir.iterdir()))} style images")
    return content_dir, style_dir


def train_samam(content_dir, style_dir):
    """Train SaMam model."""
    train_script = SAMAM_REPO / 'TRAIN' / 'train_SaMam.py'
    log_dir = CKPT_DIR / 'logs'
    log_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        PYTHON, str(train_script),
        '--gpus', '0',
        '--iterations', str(TRAIN_ITERATIONS),
        '--log-dir', str(log_dir),
        '--content', str(content_dir),
        '--style', str(style_dir),
        '--batch-size', str(BATCH_SIZE),
        '--train-image-size', str(IMAGE_SIZE),
        '--train-crop-size', str(IMAGE_SIZE),
        '--eval-image-size', str(IMAGE_SIZE),
        '--num-workers', '0',
        '--pin-memory', '0',
        '--val-interval', str(VAL_INTERVAL),
        '--precision', '32-true',  # mamba_ssm selective_scan_cuda requires float32 D param
        '--gradient-checkpointing', '1',
    ]
    print(f"  CMD: {' '.join(cmd)}")
    print(f"  Training SaMam: {TRAIN_ITERATIONS} iters, batch={BATCH_SIZE}, size={IMAGE_SIZE}")

    train_start = time.time()
    print(f"  Training started at {time.strftime('%Y-%m-%d %H:%M:%S')}")
    result = subprocess.run(cmd, cwd=str(SAMAM_REPO / 'TRAIN'))
    train_elapsed = time.time() - train_start
    print(f"  SaMam training took {train_elapsed/60:.1f} minutes (returncode={result.returncode})")

    # Find final checkpoint (train_SaMam.py saves to ./final_model.ckpt relative to cwd)
    final_ckpt = SAMAM_REPO / 'TRAIN' / 'final_model.ckpt'
    if final_ckpt.exists():
        # Save training time
        time_file = CKPT_DIR / '_TRAIN_TIME'
        time_file.write_text(f'train_min={train_elapsed/60:.1f}\n')
        return final_ckpt, train_elapsed

    # Look for checkpoint in log_dir
    ckpts = list(log_dir.rglob('*.ckpt'))
    if ckpts:
        latest = max(ckpts, key=lambda p: p.stat().st_mtime)
        return latest, train_elapsed

    return None, train_elapsed


def inference_samam(checkpoint):
    """Run SaMam inference on all 150 test images × 5 target styles."""
    SAMAM_OUT.mkdir(parents=True, exist_ok=True)

    # Prepare test data
    test_content_dir = DATA_DIR / 'test_content'
    test_style_dir = DATA_DIR / 'test_style'
    test_content_dir.mkdir(exist_ok=True)
    test_style_dir.mkdir(exist_ok=True)

    # Copy all 150 test images as content
    for style in STYLES:
        src_dir = TEST_DIR / style
        for f in sorted(src_dir.iterdir()):
            if f.suffix.lower() in ('.jpg', '.png', '.jpeg'):
                dst = test_content_dir / f'{style}__{f.name}'
                if not dst.exists():
                    shutil.copy2(str(f), str(dst))

    # Copy first test image per style as style reference
    style_refs = {}
    for style in STYLES:
        src_dir = TEST_DIR / style
        for f in sorted(src_dir.iterdir()):
            if f.suffix.lower() in ('.jpg', '.png', '.jpeg'):
                dst = test_style_dir / f'{style}__{f.name}'
                if not dst.exists():
                    shutil.copy2(str(f), str(dst))
                style_refs[style] = dst
                break

    print(f"  Content: {len(list(test_content_dir.iterdir()))} images")
    print(f"  Style refs: {len(list(test_style_dir.iterdir()))} images")

    # Run inference using test_image.py
    test_script = SAMAM_REPO / 'TEST' / 'test_image.py'
    raw_output = OUT_ROOT / 'data' / 'samam_raw_output'
    raw_output.mkdir(parents=True, exist_ok=True)

    cmd = [
        PYTHON, str(test_script),
        '--content-dir', str(test_content_dir) + '/',
        '--style-dir', str(test_style_dir) + '/',
        '--output-dir', str(raw_output),
        '--model_ckpt', str(checkpoint),
        '--save-as', 'png',
        '--style-size', str(IMAGE_SIZE),
        # Model architecture defaults match training
    ]
    print(f"  CMD: {' '.join(cmd)}")

    inf_start = time.time()
    print(f"  Inference started at {time.strftime('%Y-%m-%d %H:%M:%S')}")
    result = subprocess.run(cmd, cwd=str(SAMAM_REPO / 'TEST'))
    inf_elapsed = time.time() - inf_start
    print(f"  SaMam inference took {inf_elapsed/60:.1f} minutes (returncode={result.returncode})")

    # test_image.py outputs as {i:02}--{j:02}.png
    # where i = content index, j = style index
    # We need to map back to {src_style}__{src_stem}__to__{tgt_style}.png
    content_files = sorted(test_content_dir.iterdir())
    style_files = sorted(test_style_dir.iterdir())

    count = 0
    for i, content_file in enumerate(content_files):
        # Parse content name: {style}__{stem}.ext
        content_stem = content_file.stem  # e.g., "Early_Renaissance__artist_title"
        if '__' in content_stem:
            src_style, src_stem = content_stem.split('__', 1)
        else:
            src_style, src_stem = 'unknown', content_stem

        for j, style_file in enumerate(style_files):
            # Parse style name: {style}__{stem}.ext
            style_stem = style_file.stem
            if '__' in style_stem:
                tgt_style, _ = style_stem.split('__', 1)
            else:
                tgt_style = style_stem

            out_name = f'{src_style}__{src_stem}__to__{tgt_style}.png'
            out_path = SAMAM_OUT / out_name

            raw_name = f'{i:02}--{j:02}.png'
            raw_path = raw_output / raw_name

            if raw_path.exists() and not out_path.exists():
                from PIL import Image
                img = Image.open(str(raw_path)).convert('RGB')
                img.save(str(out_path))
                count += 1

    # Record inference time
    time_file = CKPT_DIR / '_INF_TIME'
    time_file.write_text(f'inf_min={inf_elapsed/60:.1f}\n')

    print(f"  Generated {count} images")
    if count > 0:
        mark_done(count)
    return count


def main():
    print("=" * 60)
    print("SaMam Reproduction (WSL mamba-ssm)")
    print(f"Time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    if is_done():
        print("[samam] Already done, skipping")
        return

    SAMAM_OUT.mkdir(parents=True, exist_ok=True)
    CKPT_DIR.mkdir(parents=True, exist_ok=True)

    # Check for existing checkpoint from our previous training (NOT the repo's pre-existing final_model.ckpt)
    train_done = CKPT_DIR / '_TRAIN_TIME'
    checkpoint = None
    if train_done.exists():
        # Look for checkpoint saved by our training (in TRAIN/final_model.ckpt or log_dir)
        our_ckpt = SAMAM_REPO / 'TRAIN' / 'final_model.ckpt'
        if our_ckpt.exists():
            # Verify it's ours by checking mtime is after train_done creation
            if our_ckpt.stat().st_mtime > train_done.stat().st_mtime - 60:
                checkpoint = our_ckpt
                print(f"  Using our trained checkpoint: {checkpoint}")
        if checkpoint is None:
            ckpts = list((CKPT_DIR / 'logs').rglob('*.ckpt')) if (CKPT_DIR / 'logs').exists() else []
            if ckpts:
                checkpoint = max(ckpts, key=lambda p: p.stat().st_mtime)
                print(f"  Using latest checkpoint from logs: {checkpoint}")

    if checkpoint is None:
        print("\n[1] Preparing data...")
        content_dir, style_dir = prepare_data()

        print("\n[2] Training SaMam...")
        checkpoint, train_min = train_samam(content_dir, style_dir)
        if checkpoint is None:
            print("  ERROR: No checkpoint produced, aborting")
            return

    print(f"\n[3] Running inference with {checkpoint}...")
    count = inference_samam(checkpoint)

    print("\n" + "=" * 60)
    print(f"SaMam complete: {count} images")
    print(f"Time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)


if __name__ == '__main__':
    main()
