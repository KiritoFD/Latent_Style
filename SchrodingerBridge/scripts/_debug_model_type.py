import sys
sys.path.insert(0, 'src')
from config_schema import load_experiment_config
from model import build_model_from_config
cfg = load_experiment_config('I:/Github/Latent_Style/SchrodingerBridge/configs/t1_asg_5ep.json')
print(f'Model contract: {cfg.model.contract_family}')
print(f'Bridge type: {getattr(cfg.bridge, "contract_family", "N/A")}')
m = build_model_from_config(cfg.model, bridge_cfg=cfg.bridge)
print(f'Model type: {type(m).__name__}')
# Check if it's SpectralODEBridge620
from spectral_bridge620 import SpectralODEBridge620
print(f'Is SpectralODEBridge620: {isinstance(m, SpectralODEBridge620)}')
# Check block types
if hasattr(m, 'blocks'):
    print(f'Block type: {type(m.blocks[0]).__name__}')
    asg = [k for k in m.blocks[0].state_dict().keys() if 'asg' in k]
    print(f'Block 0 asg keys: {asg}')
