#!/bin/bash
# Check pixel256 summary.json transfer metrics and ablation progress
PYTHON=/home/xy/venvs/samam312/bin/python

echo "=== Pixel256 summary.json transfer metrics ==="
$PYTHON -c "
import json
with open('/mnt/i/Github/Latent_Style/SchrodingerBridge/exp/pixel256_photo2art/pixel256_b1_e5_softmax/full_eval/epoch_0003/summary.json') as f:
    s = json.load(f)
a = s.get('analysis', {})
t = a.get('style_transfer_ability', {})
ap = a.get('all_pairs_overview', {})
idt = a.get('identity_reconstruction', {})
print(f'transfer: CLIP-S={t.get(\"clip_style\",0):.4f}, CLIP-T={t.get(\"clip_t\",0):.4f}, LPIPS={t.get(\"content_lpips\",0):.4f}')
print(f'allpairs: CLIP-S={ap.get(\"clip_style\",0):.4f}, CLIP-T={ap.get(\"clip_t\",0):.4f}, LPIPS={ap.get(\"content_lpips\",0):.4f}')
print(f'identity: CLIP-S={idt.get(\"clip_style\",0):.4f}, LPIPS={idt.get(\"content_lpips\",0):.4f}')
"

echo ""
echo "=== Latent256 e10 summary (for comparison) ==="
find /mnt/i/Github/Latent_Style/SchrodingerBridge/exp -name "summary.json" -path "*latent256*" 2>/dev/null | head -3
for f in $(find /mnt/i/Github/Latent_Style/SchrodingerBridge/exp -name "summary.json" -path "*latent256*" 2>/dev/null | head -1); do
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
echo "=== Ablation progress ==="
EXP_ROOT="/mnt/i/Github/Latent_Style/SchrodingerBridge/exp_ablation_620"
completed=$(ls ${EXP_ROOT}/*/full_eval/epoch_0003/summary.json 2>/dev/null | wc -l)
echo "Completed: $completed / 47"
echo "Currently running:"
ps aux | grep run_evaluation | grep -v grep | awk '{for(i=11;i<=NF;i++) if($i ~ /exp_ablation/) {print "  "$i; break}}'
