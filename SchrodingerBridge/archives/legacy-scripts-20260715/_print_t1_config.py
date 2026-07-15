"""Print T1 ASG checkpoint config."""
import torch, json

ckpt = torch.load(
    r"I:\Github\Latent_Style\SchrodingerBridge\exp\t1_asg_5ep\epoch_0005.pt",
    map_location="cpu",
    weights_only=False,
)
cfg = ckpt.get("config", ckpt.get("cfg", {}))
print(json.dumps(cfg, indent=2, default=str)[:3000])
