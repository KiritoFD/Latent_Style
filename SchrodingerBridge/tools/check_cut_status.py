"""Check CUT training and inference status for all 5 styles."""
import os
from pathlib import Path

STYLES = ['Early_Renaissance', 'Impressionism', 'Minimalism', 'Rococo', 'Ukiyo_e']
CKPT_ROOT = Path(r'I:\GitHub\Latent_Style\SchrodingerBridge\exp\baseline_v2\checkpoints\cut')
RESULTS_ROOT = Path(r'I:\GitHub\Latent_Style\SchrodingerBridge\exp\baseline_v2\data')
IMAGES_CUT = Path(r'I:\GitHub\Latent_Style\SchrodingerBridge\exp\baseline_v2\images\cut')

print("=== CUT STATUS CHECK ===")
total_fake = 0
for s in STYLES:
    ckpt_dir = CKPT_ROOT / f'cut_to_{s}'
    train_done = ckpt_dir / '_TRAIN_DONE'
    td = "YES" if train_done.exists() else "NO"

    # Count epochs
    epochs = sorted(ckpt_dir.glob('*_net_G.pth'))
    epoch_nums = [e.name.split('_')[0] for e in epochs if e.name[0].isdigit()]
    latest = epoch_nums[-1] if epoch_nums else "0"

    # Check loss_log last line
    loss_log = ckpt_dir / 'loss_log.txt'
    last_line = ""
    if loss_log.exists():
        lines = loss_log.read_text().strip().split('\n')
        last_line = lines[-1][:80] if lines else ""

    # Check fake_B count
    fake_b_dir = RESULTS_ROOT / f'cut_results_{s}' / f'cut_to_{s}' / 'test_latest' / 'images' / 'fake_B'
    fake_count = 0
    if fake_b_dir.exists():
        fake_count = len(list(fake_b_dir.glob('*.png')))
        total_fake += fake_count

    print(f"{s}: TD={td} epochs={latest}/{4} fake_B={fake_count}")
    if last_line:
        print(f"  last: {last_line}")

print(f"\nTotal fake_B images: {total_fake}")
print(f"Images in cut/: {len(list(IMAGES_CUT.glob('*.png'))) if IMAGES_CUT.exists() else 0}")
