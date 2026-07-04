"""Quick test: load VAE from local modelscope cache path."""
import sys
from pathlib import Path

SCHRODINGER_SRC = Path("/mnt/i/Github/Latent_Style/SchrodingerBridge/src")
sys.path.insert(0, str(SCHRODINGER_SRC))

from utils.inference import load_vae  # noqa: E402

LOCAL_VAE = "/mnt/i/Github/Latent_Style/SchrodingerBridge/eval_cache/hf/modelscope/stabilityai_sd-vae-ft-ema/stabilityai/sd-vae-ft-ema"
CACHE_DIR = "/mnt/i/Github/Latent_Style/SchrodingerBridge/eval_cache/hf"

print(f"[TEST] Loading VAE from: {LOCAL_VAE}")
vae = load_vae(
    device="cuda",
    model_id=LOCAL_VAE,
    cache_dir=CACHE_DIR,
    enable_xformers=False,
)
print(f"[TEST] VAE_OK type={type(vae).__name__}")
print(f"[TEST] dtype={next(vae.parameters()).dtype}")
print(f"[TEST] device={next(vae.parameters()).device}")
