import json
from pathlib import Path

# Refactored eval results
refactor_path = r'I:\Github\Latent_Style\SchrodingerBridge\exp\refactor_verify\t1_asg_5ep\summary.json'
data = json.loads(Path(refactor_path).read_text(encoding='utf-8'))
analysis = data.get('analysis', {})
all_pairs = analysis.get('all_pairs_overview', {})

print('='*70)
print('REFACTOR VERIFICATION: T1 ASG 5ep on refactored codebase')
print('='*70)
print(f"  CLIP-S:  {all_pairs.get('clip_style', 'N/A')}")
print(f"  LPIPS:   {all_pairs.get('content_lpips', 'N/A')}")

print('\n' + '='*70)
print('BASELINE (before refactor)')
print('='*70)
print('  CLIP-S:  0.7261')
print('  LPIPS:   0.3354')

# Compute delta
cs = all_pairs.get('clip_style', 0)
lp = all_pairs.get('content_lpips', 0)
print('\n' + '='*70)
print('DELTA (refactored - baseline)')
print('='*70)
print(f"  CLIP-S:  {cs - 0.7261:+.4f}  ({'PASS' if abs(cs - 0.7261) < 0.005 else 'FAIL'})")
print(f"  LPIPS:   {lp - 0.3354:+.4f}  ({'PASS' if abs(lp - 0.3354) < 0.005 else 'FAIL'})")
