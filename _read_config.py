import json
import sys

try:
    with open(r'I:\Github\Latent_Style\exp\clean_baseline\w34_clean_20260530\run_config.json', 'r') as f:
        c = json.load(f)
    
    for section, values in c.items():
        if isinstance(values, dict):
            for k, v in values.items():
                if any(x in k.lower() for x in ['high', 'transport', 'swd', 'abs']):
                    print(f"{k}: {v}")
except Exception as e:
    print(f"Error: {e}")
