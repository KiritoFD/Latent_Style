"""Check which pyiqa dependencies are available on remote."""
import importlib.util

# Core deps needed by pyiqa MUSIQ
DEPS = [
    "timm", "einops", "scipy", "cv2", "skimage", "facexlib",
    "addict", "lmdb", "bitsandbytes", "accelerate", "pandas",
    "sentencepiece", "yaml", "tqdm", "transformers", "torch",
    "torchvision", "PIL", "numpy", "future",
]
for m in DEPS:
    spec = importlib.util.find_spec(m)
    print(f"{m:20s} {'OK' if spec else 'MISSING'}")
