import json
from pathlib import Path

# Compare with T1 ASG 5ep baseline
t1_path = r'I:\Github\Latent_Style\SchrodingerBridge\exp\t1_asg_5ep\full_eval\epoch_0005\summary.json'
sf1_path = r'I:\Github\Latent_Style\SchrodingerBridge\exp\712_sf1_subband\full_eval\summary.json'

for label, path in [('T1 ASG 5ep (baseline)', t1_path), ('712 SF1 Subband', sf1_path)]:
    try:
        data = json.loads(Path(path).read_text(encoding='utf-8'))
    except Exception as e:
        print(f'{label}: cannot load - {e}')
        continue
    analysis = data.get('analysis', {})
    all_pairs = analysis.get('all_pairs_overview', {})
    print(f'\n=== {label} ===')
    print(f"  CLIP-S:  {all_pairs.get('clip_style', 'N/A')}")
    print(f"  LPIPS:   {all_pairs.get('content_lpips', 'N/A')}")
    print(f"  DINO-S:  {all_pairs.get('dino_style', 'N/A')}")
    print(f"  DINO-C:  {all_pairs.get('dino_structure', 'N/A')}")

# Also check what fields are available
data = json.loads(Path(sf1_path).read_text(encoding='utf-8'))
analysis = data.get('analysis', {})
all_pairs = analysis.get('all_pairs_overview', {})
print('\n=== Available fields in 712 SF1 all_pairs_overview ===')
for k, v in sorted(all_pairs.items()):
    print(f"  {k}: {v}")
