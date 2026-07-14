import json
from pathlib import Path

summary_path = r'I:\Github\Latent_Style\SchrodingerBridge\exp\712_sf1_subband\full_eval\summary.json'
data = json.loads(Path(summary_path).read_text(encoding='utf-8'))

print('=== 712 SF1 Subband Schedule - Evaluation Results ===')
analysis = data.get('analysis', {})

all_pairs = analysis.get('all_pairs_overview', {})
transfer = analysis.get('style_transfer_ability', {})
photo = analysis.get('photo_to_art_performance', {})
identity = analysis.get('identity_reconstruction', {})

print('\n--- All Pairs Overview ---')
print(f"  CLIP-S:        {all_pairs.get('clip_style', 'N/A')}")
print(f"  CLIP-C:        {all_pairs.get('clip_content', 'N/A')}")
print(f"  LPIPS:         {all_pairs.get('content_lpips', 'N/A')}")
print(f"  DINO-S:        {all_pairs.get('dino_style', 'N/A')}")
print(f"  DINO-C:        {all_pairs.get('dino_structure', 'N/A')}")
print(f"  CMMD:          {all_pairs.get('cmmd', 'N/A')}")

print('\n--- Style Transfer (off-diagonal) ---')
print(f"  CLIP-S:        {transfer.get('clip_style', 'N/A')}")
print(f"  CLIP-C:        {transfer.get('clip_content', 'N/A')}")
print(f"  LPIPS:         {transfer.get('content_lpips', 'N/A')}")
print(f"  DINO-S:        {transfer.get('dino_style', 'N/A')}")
print(f"  DINO-C:        {transfer.get('dino_structure', 'N/A')}")

print('\n--- Identity Reconstruction ---')
print(f"  CLIP-S:        {identity.get('clip_style', 'N/A')}")
print(f"  LPIPS:         {identity.get('content_lpips', 'N/A')}")
print(f"  DINO-C:        {identity.get('dino_structure', 'N/A')}")

# Timings
timings = data.get('timings', {})
print('\n--- Timings ---')
total = sum(timings.values()) if timings else 0
print(f"  Total wall:    {total:.1f}s")
for k, v in sorted(timings.items(), key=lambda x: -x[1])[:8]:
    print(f"  {k}: {v:.1f}s")
