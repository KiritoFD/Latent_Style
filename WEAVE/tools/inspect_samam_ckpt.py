"""Inspect SaMam checkpoint hyperparameters."""
import sys
import torch

ckpt_path = sys.argv[1] if len(sys.argv) > 1 else r"I:\Github\Latent_Style\exp_samam\training\samam_distinct5_512_scratch_7k_250eval_remote\step_checkpoints\step-step=020000.ckpt"

ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
hp = ck.get("hyper_parameters", {})
print("=== All hyper_parameters ===")
for k, v in sorted(hp.items()):
    print(f"  {k}: {v}")

# Also inspect state_dict shapes for the offending tensors
sd = ck.get("state_dict", {})
print("\n=== content_encoder.0 shapes ===")
for k in sd.keys():
    if k.startswith("content_encoder.0") or k.startswith("style_encoder.0"):
        print(f"  {k}: {tuple(sd[k].shape)}")
