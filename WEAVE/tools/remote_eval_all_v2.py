"""Batch evaluation of all baseline methods on remote server.
Uses run_evaluation.py with LPIPS-VGG (remote default).
"""
import os, sys, json, subprocess
from pathlib import Path

STYLES = 'Early_Renaissance,Impressionism,Minimalism,Rococo,Ukiyo_e'
TEST_DIR = r'I:\wikiart_distinct5_samam_512_classview\test'
EVAL_SCRIPT = r'I:\GitHub\Latent_Style\SchrodingerBridge\src\utils\run_evaluation.py'
IMAGE_ROOT = Path(r'I:\GitHub\Latent_Style\SchrodingerBridge\exp\baseline_v2\images')
EVAL_ROOT = Path(r'I:\GitHub\Latent_Style\SchrodingerBridge\exp\baseline_v2\eval')
IDT_BASELINE = '0.6399'

METHODS = [
    'identity',
    'sdedit_str010', 'sdedit_str020', 'sdedit_str035', 'sdedit_str040',
    'sdturbo',
    'adain_v32k',
    'styleid',
    'samam', 'samst', 's2wat', 'cut',
]

def eval_method(method):
    """Evaluate a single method."""
    img_dir = IMAGE_ROOT / method
    eval_dir = EVAL_ROOT / method
    
    if not img_dir.exists() or not any(img_dir.iterdir()):
        print(f"  [{method}] No images found, skipping")
        return None
    
    # Check if already done
    summary = eval_dir / 'summary.json'
    if summary.exists():
        print(f"  [{method}] Already evaluated, skipping")
        return json.loads(summary.read_text())
    
    eval_dir.mkdir(parents=True, exist_ok=True)
    
    # Copy images to eval_dir/images/ 
    images_dir = eval_dir / 'images'
    images_dir.mkdir(exist_ok=True)
    
    import shutil
    for f in img_dir.iterdir():
        if f.suffix.lower() in ('.png', '.jpg', '.jpeg') and not f.name.startswith('_'):
            dst = images_dir / f.name
            if not dst.exists():
                shutil.copy2(str(f), str(dst))
    
    cmd = [
        sys.executable, EVAL_SCRIPT,
        str(eval_dir),
        '--reuse_generated',
        '--save_generated_images',
        '--style_subdirs', STYLES,
        '--test_dir', TEST_DIR,
        '--eval_only_lpips_clip_style',
        '--clip_style_idt_baseline', IDT_BASELINE,
    ]
    
    print(f"  [{method}] Running evaluation...")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  [{method}] ERROR: {result.stderr[:500]}")
        return None
    
    if summary.exists():
        return json.loads(summary.read_text())
    return None

def main():
    results = {}
    for method in METHODS:
        print(f"\nEvaluating {method}...")
        data = eval_method(method)
        if data and 'all_pairs_overview' in data:
            ov = data['all_pairs_overview']
            results[method] = {
                'clip_style': ov.get('clip_style'),
                'content_lpips': ov.get('content_lpips'),
                'clip_s_delta_idt': ov.get('clip_s_delta_idt'),
                'clip_t': ov.get('clip_t'),
            }
            print(f"  clip_style={ov.get('clip_style'):.4f}  "
                  f"lpips={ov.get('content_lpips'):.4f}  "
                  f"delta_idt={ov.get('clip_s_delta_idt'):+.4f}")
        else:
            results[method] = None
    
    # Save unified results
    out_path = EVAL_ROOT / 'unified_results.json'
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nResults saved to {out_path}")
    
    # Print summary table
    print("\n" + "=" * 80)
    print(f"{'Method':<20} {'CLIP-S':>8} {'LPIPS':>8} {'Δ_idt':>8} {'CLIP-T':>8}")
    print("-" * 80)
    for method, r in results.items():
        if r:
            print(f"{method:<20} {r['clip_style']:8.4f} {r['content_lpips']:8.4f} "
                  f"{r['clip_s_delta_idt']:+8.4f} {r['clip_t']:8.4f}")
        else:
            print(f"{method:<20} {'N/A':>8} {'N/A':>8} {'N/A':>8} {'N/A':>8}")

if __name__ == '__main__':
    main()
