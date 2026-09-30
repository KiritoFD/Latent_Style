import torch
print("torch:", torch.__version__)
print("cuda:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device:", torch.cuda.get_device_name(0))

import transformers
print("transformers:", transformers.__version__)

from PIL import Image
print("PIL ok")

# Test accessing a file
import pathlib
p = pathlib.Path("/mnt/i/Github/Latent_Style/exp_samam/training/samam_distinct5_512_scratch_7k_250eval_remote/c/step_000250/images/Early_Renaissance__Early_Renaissance__andrea-mantegna_adoration-of-the-magi-central-panel-from-the-altarpiece__to__Early_Renaissance.png")
print(f"file exists: {p.exists()}")
img = Image.open(p).convert("RGB")
print(f"PIL OK: {img.size}")