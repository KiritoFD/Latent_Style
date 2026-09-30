"""Self-contained bf16 VAE evaluation wrapper.
Copy this to remote WEAVE root and run with: python run_eval_bf16.py --checkpoint ... --batch_size 16 ...
"""
import sys, os, importlib

# Add WEAVE to path
weave_root = os.path.dirname(os.path.abspath(__file__))
if weave_root not in sys.path:
    sys.path.insert(0, weave_root)

# ---- Force load utils.inference first, then patch ----
import utils.inference as infer_mod

import torch
_orig = infer_mod.load_vae

def _load_vae_bf16(*args, **kwargs):
    kwargs.setdefault("compile_dtype", torch.bfloat16)
    print("[bf16] VAE compile_dtype = torch.bfloat16")
    return _orig(*args, **kwargs)

infer_mod.load_vae = _load_vae_bf16
print("[bf16] Patched utils.inference.load_vae")

# ---- Now import and run evaluation ----
sys.argv = [sys.argv[0]] + sys.argv[1:]  # ensure argv is intact
import utils.run_evaluation as eval_mod
eval_mod.main()