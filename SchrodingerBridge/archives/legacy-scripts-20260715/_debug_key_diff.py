import sys, torch
sys.path.insert(0, 'src')
from config_schema import load_experiment_config
from model import build_model_from_config

cfg = load_experiment_config('I:/Github/Latent_Style/SchrodingerBridge/configs/t1_asg_5ep.json')
model = build_model_from_config(cfg.model, bridge_cfg=cfg.bridge)
ckpt = torch.load('I:/Github/Latent_Style/SchrodingerBridge/exp/t1_asg_5ep/epoch_0005.pt', map_location='cpu', weights_only=False)
sd = ckpt.get('model_state_dict', ckpt)

model_keys = set(model.state_dict().keys())
ckpt_keys = set(sd.keys())

model_asg = sorted([k for k in model_keys if 'asg' in k])
ckpt_asg = sorted([k for k in ckpt_keys if 'asg' in k])
print(f'Model asg keys ({len(model_asg)}): {model_asg[:6]}')
print(f'Ckpt asg keys ({len(ckpt_asg)}): {ckpt_asg[:6]}')
print(f'In ckpt not model: {sorted(ckpt_keys - model_keys)[:10]}')
print(f'In model not ckpt: {sorted(model_keys - ckpt_keys)[:10]}')
# Check if there's a prefix difference
print(f'\nModel has blocks.0.asg_proj: {"blocks.0.asg_proj.weight" in model_keys}')
print(f'Ckpt has blocks.0.asg_proj: {"blocks.0.asg_proj.weight" in ckpt_keys}')
# Maybe model has it under different prefix
model_block0 = sorted([k for k in model_keys if 'blocks.0' in k])
print(f'\nAll model blocks.0 keys: {model_block0}')
