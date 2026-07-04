import sys
sys.path.insert(0, 'src')
from config_schema import load_experiment_config
for name in ['620_ablation_dino_baseline_smoke','620_ablation_dino_adapter_smoke','620_ablation_intrinsic_latent_smoke']:
    cfg = load_experiment_config(f'exp/620_spatial_bridge/{name}/config.json')
    print(name, 'condition_source=', cfg.model.style_condition_source, 'adapter=', cfg.model.style_dino_adapter_enabled)
