import sys, torch
sys.path.insert(0, 'src')
from config_schema import load_experiment_config
from model import build_model_from_config
from spectral_losses620 import SpectralODEObjective620

cfg = load_experiment_config('I:/Github/Latent_Style/SchrodingerBridge/configs/t1_asg_5ep.json')
model = build_model_from_config(cfg.model, bridge_cfg=cfg.bridge).cuda().eval()
ckpt = torch.load('I:/Github/Latent_Style/SchrodingerBridge/exp/t1_asg_5ep/epoch_0005.pt', map_location='cpu', weights_only=False)
sd = ckpt.get('model_state_dict', ckpt)
missing, unexpected = model.load_state_dict(sd, strict=False)
print(f'Load: missing={len(missing)} unexpected={len(unexpected)}')

# Test forward pass
loss_fn = SpectralODEObjective620(cfg)
content = torch.randn(2, 4, 32, 32, device='cuda')
target_style = torch.randn(2, 4, 32, 32, device='cuda')
target_style_id = torch.tensor([0, 1], device='cuda')
with torch.no_grad():
    metrics = loss_fn.compute(model, content=content, target_style=target_style, target_style_id=target_style_id)
print(f'Forward OK: loss={metrics["loss"].item():.4f} flow={metrics["flow"].item():.4f}')
print(f'  contrastive_swd={metrics["loss_contrastive_swd"].item():.4f} (should be 0, disabled)')
print('ALL TESTS PASSED')
