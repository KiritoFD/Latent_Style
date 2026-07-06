#!/usr/bin/env bash
set -uo pipefail
echo "===ablation_620/infra_I0_baseline config (data section)==="
PYTHON=/home/xy/venvs/samam312/bin/python
$PYTHON -c "
import json
with open('/mnt/i/Github/Latent_Style/SchrodingerBridge/ablation_620/infra_I0_baseline/config.json') as f:
    cfg = json.load(f)
print('===MODEL===')
for k, v in cfg.get('model', {}).items():
    print(f'  {k}: {v}')
print('===DATA===')
for k, v in cfg.get('data', {}).items():
    print(f'  {k}: {v}')
print('===TRAINING===')
for k, v in cfg.get('training', {}).items():
    if k in ['seed','batch_size','num_epochs','learning_rate','test_image_dir','full_eval_each_epoch','full_eval_defer_until_training_end','full_eval_batch_size','full_eval_vae_decode_batch_size','full_eval_cache_dir']:
        print(f'  {k}: {v}')
print('===CHECKPOINT===')
for k, v in cfg.get('checkpoint', {}).items():
    print(f'  {k}: {v}')
" 2>/dev/null

echo ""
echo "===infra_I0_baseline (saved in exp_ablation_620)==="
$PYTHON -c "
import json
with open('/mnt/i/Github/Latent_Style/SchrodingerBridge/exp_ablation_620/infra_I0_baseline/config.json') as f:
    cfg = json.load(f)
print('===MODEL===')
for k in ['contract_family','latent_channels','num_styles','base_dim','num_res_blocks']:
    print(f'  {k}: {cfg[\"model\"].get(k)}')
print('===DATA===')
for k in ['data_root','style_subdirs','latent_cache_dir','latent_cache_mode','dino_cache_path','pairing_cache_path']:
    print(f'  {k}: {cfg[\"data\"].get(k)}')
print('===TRAINING===')
for k in ['seed','batch_size','num_epochs','learning_rate','test_image_dir','full_eval_each_epoch','full_eval_defer_until_training_end']:
    print(f'  {k}: {cfg[\"training\"].get(k)}')
print('===CHECKPOINT===')
print(f'  save_dir: {cfg[\"checkpoint\"].get(\"save_dir\")}')
" 2>/dev/null

echo ""
echo "===Check if wikiart_distinct5_samam_512_latents_ema exists==="
ls /mnt/i/wikiart_distinct5_samam_512_latents_ema/ 2>/dev/null | head -5
ls /mnt/i/wikiart_distinct5_samam_512_latents_ema/train/ 2>/dev/null | head -10
ls /mnt/i/wikiart_distinct5_samam_512_latents_ema/train/.latent_cache/ 2>/dev/null | head -5

echo ""
echo "===Check test_image_dir resolves==="
ls /mnt/i/Github/Latent_Style/SchrodingerBridge/style_data/ 2>/dev/null | head -5
ls /mnt/i/Github/Latent_Style/style_data/ 2>/dev/null | head -5
ls -d /mnt/i/wikiart_distinct5_samam_512_classview/test 2>/dev/null

echo ""
echo "===List one ablation exp_ablation_620/DA01_backbone1/config.json (data section)==="
$PYTHON -c "
import json
with open('/mnt/i/Github/Latent_Style/SchrodingerBridge/exp_ablation_620/DA01_backbone1/config.json') as f:
    cfg = json.load(f)
print('===DATA===')
for k in ['data_root','style_subdirs','latent_cache_dir','latent_cache_mode','dino_cache_path','pairing_cache_path']:
    print(f'  {k}: {cfg[\"data\"].get(k)}')
print('===TRAINING===')
for k in ['batch_size','num_epochs','test_image_dir','full_eval_each_epoch','full_eval_defer_until_training_end']:
    print(f'  {k}: {cfg[\"training\"].get(k)}')
print('===CHECKPOINT===')
print(f'  save_dir: {cfg[\"checkpoint\"].get(\"save_dir\")}')
" 2>/dev/null
