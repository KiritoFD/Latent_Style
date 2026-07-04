#!/usr/bin/env bash
echo "=== Find adain/wct/samst on photo2art 5 styles ==="
find /mnt/i -maxdepth 6 -type d -name "cezanne" 2>/dev/null | head -10
echo ""
echo "=== Check exp_baselines dir ==="
ls /mnt/i/Github/Latent_Style/exp_baselines/ 2>/dev/null | head -30
echo ""
echo "=== Check baseline_v2/eval/seedream ==="
ls /mnt/i/Github/Latent_Style/SchrodingerBridge/exp/baseline_v2/eval/seedream/ 2>/dev/null
echo ""
echo "=== Check baseline_v2/eval/seedream/images filenames ==="
find /mnt/i/Github/Latent_Style/SchrodingerBridge/exp/baseline_v2/eval/seedream/ -name "*.png" -o -name "*.jpg" 2>/dev/null | head -5
echo ""
echo "=== Search for any baseline output with _to_cz/_to_monet/_to_Hayao/_to_vangogh ==="
find /mnt/i -maxdepth 7 -name "*_to_cz*.png" -o -name "*_to_monet*.png" -o -name "*_to_Hayao*.png" 2>/dev/null | head -10
