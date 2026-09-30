import os
import sys
import shutil
from pathlib import Path

# Enable compile debug logs and dump generated Triton code
os.environ["TORCH_COMPILE_DEBUG"] = "1"
os.environ["TORCHDYNAMO_VERBOSE"] = "1"

import torch
import torch._inductor.config as inductor_config

# Enable same config options
try:
    inductor_config.coordinate_descent_tuning = True
except AttributeError:
    pass
try:
    inductor_config.triton.autotune_cublasLt = True
except AttributeError:
    pass
try:
    inductor_config.freezing = True
except AttributeError:
    pass
try:
    inductor_config.triton.cudagraphs = True
    inductor_config.triton.cudagraph_trees = False
except AttributeError:
    pass

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.inference import VAEDecodeWrapper, download_vae_with_fallback

DEVICE = "cuda"

def main():
    print("Loading VAE decoder...")
    vae = download_vae_with_fallback("ema", device=DEVICE)
    vae = vae.to(device=DEVICE, dtype=torch.float16, memory_format=torch.channels_last)
    vae.eval()
    vae.requires_grad_(False)
    
    decoder = VAEDecodeWrapper(vae).to(device=DEVICE, dtype=torch.float16, memory_format=torch.channels_last)
    decoder.eval()
    decoder.requires_grad_(False)
    
    # Configure fresh debug output directory
    debug_dir = Path("torch_compile_debug")
    if debug_dir.exists():
        shutil.rmtree(debug_dir)
        
    print("Compiling model (TORCH_COMPILE_DEBUG=1)...")
    compiled = torch.compile(decoder, mode="max-autotune", fullgraph=True, dynamic=False)
    
    dummy = torch.randn(8, 4, 64, 64, device=DEVICE, dtype=torch.float16).contiguous(memory_format=torch.channels_last)
    
    with torch.inference_mode():
        _ = compiled(dummy)
    torch.cuda.synchronize()
    
    print("Compilation and warmup run complete.")
    print(f"Debug files are dumped to: {debug_dir.resolve()}")

if __name__ == "__main__":
    main()
