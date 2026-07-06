#!/usr/bin/env bash
SUMMARY=/mnt/i/Github/Latent_Style/SchrodingerBridge/exp/pixel256_photo2art/pixel256_b1_e5_softmax/full_eval/epoch_0003/summary.json
/home/xy/venvs/samam312/bin/python -c "
import json
with open('$SUMMARY') as f:
    s = json.load(f)
a = s.get('analysis', {})
t = a.get('style_transfer_ability', {})
ap = a.get('all_pairs_overview', {})
idt = a.get('identity_reconstruction', {})
print('=== Pixel256 epoch_0003 Results ===')
print(f'transfer CLIP-S: {t.get(\"clip_style\", \"N/A\")}')
print(f'transfer CLIP-T: {t.get(\"clip_t\", \"N/A\")}')
print(f'transfer LPIPS:  {t.get(\"content_lpips\", \"N/A\")}')
print(f'all_pairs CLIP-S: {ap.get(\"clip_style\", \"N/A\")}')
print(f'all_pairs CLIP-T: {ap.get(\"clip_t\", \"N/A\")}')
print(f'all_pairs LPIPS:  {ap.get(\"content_lpips\", \"N/A\")}')
print(f'identity CLIP-S:  {idt.get(\"clip_style\", \"N/A\")}')
print(f'identity LPIPS:   {idt.get(\"content_lpips\", \"N/A\")}')
# Check for MUSIQ and ART-FID
musiq = a.get('musiq_overview', a.get('musiq', {}))
artfid = a.get('art_fid_overview', a.get('art_fid', {}))
print(f'MUSIQ: {musiq if musiq else \"N/A\"}')
print(f'ART-FID: {artfid if artfid else \"N/A\"}')
print()
print('=== Full analysis keys ===')
for k in a:
    print(f'  {k}')
"
