"""SaMST training + inference on distinct5_512 - run on remote server.
Uses train2 mode (pre-computed Gram matrices) for faster training.
"""
import os, sys, json, shutil, subprocess
from pathlib import Path

# ── Config ──
STYLES = ['Early_Renaissance', 'Impressionism', 'Minimalism', 'Rococo', 'Ukiyo_e']
SAMST_REPO = Path(r'I:\GitHub\Latent_Style\Related_Works\repos\SaMST-main')
TRAIN_DIR = Path(r'I:\wikiart_distinct5_samam_512_classview\train')
TEST_DIR = Path(r'I:\wikiart_distinct5_samam_512_classview\test')
OUT_ROOT = Path(r'I:\GitHub\Latent_Style\SchrodingerBridge\exp\baseline_v2')
IMAGE_OUT = OUT_ROOT / 'images' / 'samst'
EVAL_OUT = OUT_ROOT / 'eval' / 'samst'
TRAIN_LOG = OUT_ROOT / 'train_logs' / 'samst'

def prepare_data():
    """Prepare ImageFolder structure for SaMST train2."""
    # SaMST train2 expects:
    # - dataset: ImageFolder format (class subfolders)
    # - style_image: directory of style images
    
    # Content images: use all train images as ImageFolder
    # Style images: same train images
    
    # Create ImageFolder structure
    content_dir = OUT_ROOT / 'data' / 'samst_content'
    content_dir.mkdir(parents=True, exist_ok=True)
    
    for style in STYLES:
        src = TRAIN_DIR / style
        dst = content_dir / style
        if dst.exists() and len(list(dst.iterdir())) > 0:
            print(f"  {style}: already linked")
            continue
        dst.mkdir(exist_ok=True)
        # Copy images
        for f in src.iterdir():
            if f.suffix.lower() in ('.jpg', '.png', '.jpeg'):
                shutil.copy2(str(f), str(dst / f.name))
        print(f"  {style}: copied {len(list(dst.iterdir()))} images")

    # Create style directories (one per style for style reference)
    for style in STYLES:
        style_dir = OUT_ROOT / 'data' / f'samst_style_{style}'
        style_dir.mkdir(parents=True, exist_ok=True)
        if len(list(style_dir.iterdir())) > 0:
            continue
        src = TRAIN_DIR / style
        for f in src.iterdir():
            if f.suffix.lower() in ('.jpg', '.png', '.jpeg'):
                shutil.copy2(str(f), str(style_dir / f.name))
    
    return content_dir

def write_train_config(content_dir):
    """Write train2.yml config file."""
    config = f"""dataset: {content_dir}
style_image: {OUT_ROOT / 'data' / 'samst_style_Early_Renaissance'}
image_size: 512
style_size: 512
batch_size: 2
epochs: 20
lr: 0.001
save_model_dir: {TRAIN_LOG}
"""
    config_path = SAMST_REPO / 'train_model' / 'train2' / 'train_distinct5.yml'
    config_path.write_text(config)
    print(f"Config written to {config_path}")
    return config_path

def train(config_path):
    """Run SaMST training."""
    train2_dir = SAMST_REPO / 'train_model' / 'train2'
    cmd = f'cd /d {train2_dir} && python train.py --config {config_path}'
    print(f"Training: {cmd}")
    result = subprocess.run(cmd, shell=True, capture_output=False)
    return result.returncode

def inference():
    """Run SaMST inference on test set."""
    # Find the latest checkpoint
    ckpt_dir = TRAIN_LOG
    checkpoints = sorted(ckpt_dir.glob('*.pth'))
    if not checkpoints:
        print("ERROR: No checkpoints found!")
        return
    
    latest_ckpt = checkpoints[-1]
    print(f"Using checkpoint: {latest_ckpt}")

    # Use SaMST test script
    # SaMST needs to be configured for 5 styles
    # Run inference for each target style
    test_script = SAMST_REPO / 'test_model' / 'test' / 'test.py'
    
    IMAGE_OUT.mkdir(parents=True, exist_ok=True)
    
    # SaMST produces one model per style, need to run for each
    # Actually SaMST train2 trains a single multi-style model
    # Check if there's a multi-style test script
    
    # For now, use a simple Python inference approach
    cmd = [
        sys.executable, str(SAMST_REPO / 'test_model' / 'test' / 'test.py'),
        '--content_dir', str(TEST_DIR),
        '--style_dir', str(TEST_DIR),
        '--output_dir', str(IMAGE_OUT),
        '--model_path', str(latest_ckpt),
    ]
    print(f"Inference: {' '.join(cmd)}")
    # This needs more work based on SaMST's actual inference interface

if __name__ == '__main__':
    print("=" * 60)
    print("SaMST Training + Inference on distinct5_512")
    print("=" * 60)
    
    content_dir = prepare_data()
    config_path = write_train_config(content_dir)
    # train(config_path)
    # inference()
    print("Setup complete. Run training manually:")
    print(f"  cd /d {SAMST_REPO / 'train_model' / 'train2'}")
    print(f"  python train.py --config {config_path}")
