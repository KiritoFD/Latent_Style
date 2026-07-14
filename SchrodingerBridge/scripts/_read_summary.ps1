$summaryPath = "I:\Github\Latent_Style\SchrodingerBridge\exp\model_probe\target_hf_subband_ft6\full_eval\adain20\summary.json"
$py = "C:\Program Files\Python312\python.exe"
$script = @"
import json
with open(r'$summaryPath', 'r') as f:
    data = json.load(f)
# Print top-level keys
print('TOP KEYS:', list(data.keys())[:20])
# Try common summary locations
for key in ['summary', 'overall', 'metrics', 'aggregate', 'mean']:
    if key in data:
        print(f'{key}:', json.dumps(data[key], indent=2)[:2000])
# If it's a list, print first and last
if isinstance(data, list):
    print('LIST len:', len(data))
    print('FIRST:', json.dumps(data[0], indent=2)[:1000])
    print('LAST:', json.dumps(data[-1], indent=2)[:1000])
# Print any key containing 'clip' or 'lpips'
for k, v in data.items():
    if 'clip' in k.lower() or 'lpips' in k.lower():
        print(f'{k}:', v if not isinstance(v, (list, dict)) else json.dumps(v)[:500])
"
& $py -c $script
