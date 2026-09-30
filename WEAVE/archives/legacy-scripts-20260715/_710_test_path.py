"""Test if Windows Python can access I: drive."""
import os
from pathlib import Path

test_dir = Path(r"I:\datasets\wikiart_distinct5_samam_512_classview\test")
print(f"test_dir exists: {test_dir.exists()}")
if test_dir.exists():
    styles = sorted([d.name for d in test_dir.iterdir() if d.is_dir()])
    print(f"Styles: {styles}")
    er = test_dir / "Early_Renaissance"
    if er.exists():
        files = sorted(os.listdir(er))[:3]
        print(f"Early_Renaissance files (first 3): {files}")

# Also check eval dir
eval_dir = Path(r"I:\Github\Latent_Style\SchrodingerBridge\exp\710_b0_t11")
print(f"\neval_dir exists: {eval_dir.exists()}")
ckpt = eval_dir / "epoch_0005.pt"
print(f"checkpoint exists: {ckpt.exists()}, size={ckpt.stat().st_size if ckpt.exists() else 'N/A'}")
