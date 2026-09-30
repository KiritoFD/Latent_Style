"""Wrapper to run run_evaluation.py with bf16 VAE."""
import sys
import os

# Monkey-patch: force bf16 VAE
sys.path.insert(0, r'I:\Github\Latent_Style\WEAVE')
os.chdir(r'I:\Github\Latent_Style\WEAVE')

import torch
import utils.inference as inference_mod

# Store original
_original_load_vae = inference_mod.load_vae

def _load_vae_bf16(*args, **kwargs):
    kwargs.setdefault('compile_dtype', torch.bfloat16)
    print("[bf16 wrapper] forcing VAE compile_dtype = torch.bfloat16")
    return _original_load_vae(*args, **kwargs)

inference_mod.load_vae = _load_vae_bf16

# Now run the original evaluation
import utils.run_evaluation as eval_mod
eval_mod.main()