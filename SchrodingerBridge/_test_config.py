import sys
sys.path.insert(0, r"I:\Github\Latent_Style\SchrodingerBridge\src")
from config_schema import load_experiment_config
try:
    cfg = load_experiment_config(r"I:\Github\Latent_Style\SchrodingerBridge\configs\630_remote_t11_long30ep.json")
    print("CONFIG OK:", cfg.ablation.name if hasattr(cfg, 'ablation') else "loaded")
except Exception as e:
    print("CONFIG ERROR:", e)
