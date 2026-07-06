#!/bin/bash
PYTHON=/home/xy/venvs/samam312/bin/python

echo "=== Latent256 e10 epoch_0010 summary ==="
$PYTHON -c "
import json
f = '/mnt/i/Github/Latent_Style/SchrodingerBridge/exp/latent256_photo2art/latent256_b16_e10/full_eval/epoch_0010/summary.json'
with open(f) as fh:
    s = json.load(fh)
a = s.get('analysis', {})
t = a.get('style_transfer_ability', {})
ap = a.get('all_pairs_overview', {})
idt = a.get('identity_reconstruction', {})
print(f'transfer: CLIP-S={t.get(\"clip_style\",0):.4f}, CLIP-T={t.get(\"clip_t\",0):.4f}, LPIPS={t.get(\"content_lpips\",0):.4f}')
print(f'allpairs: CLIP-S={ap.get(\"clip_style\",0):.4f}, CLIP-T={ap.get(\"clip_t\",0):.4f}, LPIPS={ap.get(\"content_lpips\",0):.4f}')
print(f'identity: CLIP-S={idt.get(\"clip_style\",0):.4f}, LPIPS={idt.get(\"content_lpips\",0):.4f}')
"

echo ""
echo "=== Check 512 main WD-VF summary for comparison ==="
find /mnt/i/Github/Latent_Style/SchrodingerBridge/exp -name "summary.json" -path "*630_phase4i2b*" 2>/dev/null | head -3
for f in $(find /mnt/i/Github/Latent_Style/SchrodingerBridge/exp -name "summary.json" -path "*630_phase4i2b*" 2>/dev/null | head -1); do
    echo "File: $f"
    $PYTHON -c "
import json
with open('$f') as fh:
    s = json.load(fh)
a = s.get('analysis', {})
t = a.get('style_transfer_ability', {})
ap = a.get('all_pairs_overview', {})
idt = a.get('identity_reconstruction', {})
print(f'transfer: CLIP-S={t.get(\"clip_style\",0):.4f}, CLIP-T={t.get(\"clip_t\",0):.4f}, LPIPS={t.get(\"content_lpips\",0):.4f}')
print(f'allpairs: CLIP-S={ap.get(\"clip_style\",0):.4f}, CLIP-T={ap.get(\"clip_t\",0):.4f}, LPIPS={ap.get(\"content_lpips\",0):.4f}')
print(f'identity: CLIP-S={idt.get(\"clip_style\",0):.4f}, LPIPS={idt.get(\"content_lpips\",0):.4f}')
"
done

echo ""
echo "=== Check 620 baseline summary (for ablation baseline) ==="
find /mnt/i/Github/Latent_Style/SchrodingerBridge/exp -name "summary.json" -path "*620*" 2>/dev/null | head -5
echo "---"
find /mnt/i/Github/Latent_Style/SchrodingerBridge/exp_ablation_620 -name "summary.json" -path "*DA01*" 2>/dev/null

echo ""
echo "=== DA01_backbone1 (ablation baseline) config ==="
$PYTHON -c "
import json
with open('/mnt/i/Github/Latent_Style/SchrodingerBridge/exp_ablation_620/DA01_backbone1/config.json') as f:
    c = json.load(f)
m = c.get('model', {})
t = c.get('training', {})
b = c.get('bridge', {})
print(f'backbone_depth={m.get(\"backbone_depth\")}, dim={m.get(\"dim\")}, num_heads={m.get(\"num_heads\")}')
print(f'batch_size={t.get(\"batch_size\")}, max_epochs={t.get(\"max_epochs\")}, test_image_dir={t.get(\"test_image_dir\")}')
print(f'objective_mode={b.get(\"objective_mode\")}')
print(f'style_attn_mode={m.get(\"style_attn_mode\")}')
"
