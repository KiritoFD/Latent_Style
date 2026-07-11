import sys, torch
sys.path.insert(0, 'src')
ckpt = torch.load('I:/Github/Latent_Style/SchrodingerBridge/exp/t1_asg_5ep/epoch_0005.pt', map_location='cpu', weights_only=False)
sd = ckpt.get('model_state_dict', ckpt)
asg_keys = [k for k in sd.keys() if 'asg' in k]
print(f'asg keys in checkpoint: {len(asg_keys)}')
for k in asg_keys[:20]:
    print(f'  {k}: shape={sd[k].shape}')
# Check if model has these keys
from config_schema import load_experiment_config
from model import build_model_from_config
cfg = load_experiment_config('I:/Github/Latent_Style/SchrodingerBridge/configs/t1_asg_5ep.json')
model = build_model_from_config(cfg.model, bridge_cfg=cfg.bridge)
model_keys = set(model.state_dict().keys())
ckpt_keys = set(sd.keys())
print(f'\nModel has asg keys: {sorted([k for k in model_keys if "asg" in k])[:10]}')
print(f'Ckpt has asg keys: {sorted([k for k in ckpt_keys if "asg" in k])[:10]}')
print(f'\nIn ckpt but not model: {sorted(ckpt_keys - model_keys)[:20]}')
