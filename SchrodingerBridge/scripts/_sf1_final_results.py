import json
from pathlib import Path

# 712 SF1 results
sf1_summary = json.loads(Path(r'I:\Github\Latent_Style\SchrodingerBridge\exp\712_sf1_subband\full_eval\summary.json').read_text(encoding='utf-8'))
sf1_dino = json.loads(Path(r'I:\Github\Latent_Style\SchrodingerBridge\exp\712_sf1_subband\full_eval\dino_summary.json').read_text(encoding='utf-8'))

sf1_all = sf1_summary.get('analysis', {}).get('all_pairs_overview', {})
sf1_transfer = sf1_summary.get('analysis', {}).get('style_transfer_ability', {})

print('='*70)
print('712 SF1 Subband Schedule (时频耦合流形调度) - 完整指标')
print('='*70)
print('\n--- All Pairs (750 samples, 全部含对角) ---')
print(f"  CLIP-S:   {sf1_all.get('clip_style', 'N/A'):.4f}")
print(f"  LPIPS:    {sf1_all.get('content_lpips', 'N/A'):.4f}")
print(f"  DINO-S:   {sf1_dino.get('all_dino_s', 'N/A'):.4f}")
print(f"  DINO-C:   {sf1_dino.get('all_dino_c', 'N/A'):.4f}")

print('\n--- Off-diagonal (600 samples, src!=tgt, 迁移能力) ---')
print(f"  CLIP-S:   {sf1_transfer.get('clip_style', 'N/A'):.4f}")
print(f"  LPIPS:    {sf1_transfer.get('content_lpips', 'N/A'):.4f}")
print(f"  DINO-S:   {sf1_dino.get('off_dino_s', 'N/A'):.4f}")
print(f"  DINO-C:   {sf1_dino.get('off_dino_c', 'N/A'):.4f}")

# Baseline T1 ASG 5ep - from memory
print('\n' + '='*70)
print('对比: T1 ASG 5ep (baseline, 当前最优)')
print('='*70)
print('  CLIP-S:   0.7261')
print('  LPIPS:    0.3354')
print('  DINO-S:   0.4843')
print('  DINO-C:   0.7692')

print('\n' + '='*70)
print('Delta (SF1 - T1)')
print('='*70)
sf1_cs = sf1_all.get('clip_style', 0)
sf1_lp = sf1_all.get('content_lpips', 0)
sf1_ds = sf1_dino.get('all_dino_s', 0)
sf1_dc = sf1_dino.get('all_dino_c', 0)
print(f"  CLIP-S:   {sf1_cs - 0.7261:+.4f}  ({'↑' if sf1_cs > 0.7261 else '↓'} {'改善' if sf1_cs > 0.7261 else '退化'})")
print(f"  LPIPS:    {sf1_lp - 0.3354:+.4f}  ({'↑' if sf1_lp > 0.3354 else '↓'} {'退化' if sf1_lp > 0.3354 else '改善'} (LPIPS低好))")
print(f"  DINO-S:   {sf1_ds - 0.4843:+.4f}  ({'↑' if sf1_ds > 0.4843 else '↓'} {'改善' if sf1_ds > 0.4843 else '退化'})")
print(f"  DINO-C:   {sf1_dc - 0.7692:+.4f}  ({'↑' if sf1_dc > 0.7692 else '↓'} {'改善' if sf1_dc > 0.7692 else '退化'})")
