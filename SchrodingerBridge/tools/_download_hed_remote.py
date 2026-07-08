"""Download HED model for StyleShot."""
import os
os.makedirs(r"C:\Users\Administrator\StyleShot\annotator\ckpts", exist_ok=True)
from huggingface_hub import hf_hub_download
f = hf_hub_download("lllyasviel/Annotators", "ControlNetHED.pth",
                     local_dir=r"C:\Users\Administrator\StyleShot\annotator\ckpts")
print(f"HED: {f}")
