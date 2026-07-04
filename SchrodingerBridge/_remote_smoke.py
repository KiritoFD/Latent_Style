"""Remote smoke test: verify synced code loads correctly."""
import sys
sys.path.insert(0, 'src')
from spectral_losses620 import SpectralODEObjective620
from spectral_bridge620 import SpectralODEBridge620
from config_schema import load_experiment_config

cfg = load_experiment_config('configs/clean_base_v2.json')
model = SpectralODEBridge620(cfg.model, cfg.bridge)
loss_fn = SpectralODEObjective620(cfg)
n_params = sum(p.numel() for p in model.parameters())
print(f'OK params={n_params:,}')
