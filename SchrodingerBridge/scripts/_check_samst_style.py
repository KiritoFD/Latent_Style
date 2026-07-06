"""Check SaMST checkpoint style count and inspect existing 256 image names."""
import torch
from pathlib import Path

# Load checkpoint
ckpt_path = r"I:\Github\Latent_Style\Related_Works\repos\SaMST-main\checkpoint\epoch_20.model"
sd = torch.load(ckpt_path, map_location="cpu", weights_only=False)

# Count style_para_list entries
style_keys = [k for k in sd.keys() if "style_para_list" in k and k.endswith(".params")]
style_keys_sorted = sorted(style_keys, key=lambda k: int(k.split(".")[2]))
print(f"style_para_list count: {len(style_keys_sorted)}")
for k in style_keys_sorted:
    shape = sd[k].shape if hasattr(sd[k], "shape") else "N/A"
    print(f"  {k}: shape={shape}")

# Sample some existing 256 images to see naming pattern
img_dir = Path(r"I:\exp_256_photo2art\samst_256\images")
if img_dir.exists():
    files = sorted(img_dir.glob("*.png"))[:5]
    print(f"\nFirst 5 images from {img_dir}:")
    for f in files:
        print(f"  {f.name}")
