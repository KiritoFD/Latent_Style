import sys, torch
sys.path.insert(0, 'src')
from config_schema import load_experiment_config
from model import build_model_from_config

cfg = load_experiment_config('I:/Github/Latent_Style/SchrodingerBridge/configs/t1_asg_5ep.json')
model = build_model_from_config(cfg.model, bridge_cfg=cfg.bridge)
ckpt = torch.load('I:/Github/Latent_Style/SchrodingerBridge/exp/t1_asg_5ep/epoch_0005.pt', map_location='cpu', weights_only=False)
sd = ckpt.get('model_state_dict', ckpt)
missing, unexpected = model.load_state_dict(sd, strict=False)
print(f'missing={len(missing)} unexpected={len(unexpected)}')
if missing:
    print('missing[:10]=', missing[:10])
if unexpected:
    print('unexpected[:10]=', unexpected[:10])
print('Checkpoint load test PASSED' if len(missing) == 0 and len(unexpected) == 0 else 'Checkpoint load test COMPLETED with warnings')
