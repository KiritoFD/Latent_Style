import torch

ckpt = torch.load(r"I:\Github\Latent_Style\SchrodingerBridge\exp\t1_asg_5ep\epoch_0005.pt", map_location="cpu", weights_only=False)
state_dict = ckpt.get("model_state_dict", ckpt.get("state_dict", {}))
for i in range(4):
    w = state_dict[f"blocks.{i}.asg_proj.weight"]
    b = state_dict[f"blocks.{i}.asg_proj.bias"]
    print(f"Block {i}: weight abs_max={w.abs().max().item():.6f}, bias={b.item():.6f}")
