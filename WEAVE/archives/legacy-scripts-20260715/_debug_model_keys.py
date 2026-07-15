import sys
sys.path.insert(0, 'src')
from config_schema import load_experiment_config
from model import build_model_from_config
cfg = load_experiment_config('I:/Github/Latent_Style/SchrodingerBridge/configs/t1_asg_5ep.json')
m = build_model_from_config(cfg.model, bridge_cfg=cfg.bridge)
print(f'Model type: {type(m).__name__}')
asg_keys = [k for k in m.state_dict().keys() if 'asg' in k]
print(f'asg keys in model: {len(asg_keys)}')
print(f'asg keys: {asg_keys[:10]}')
block0_keys = [k for k in m.state_dict().keys() if k.startswith('blocks.0.')]
print(f'blocks.0.* keys ({len(block0_keys)}): {block0_keys[:15]}')
