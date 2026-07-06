#!/bin/bash
PYTHON=/home/xy/venvs/samam312/bin/python
echo "===EVAL SETTINGS==="
$PYTHON -c "
import json
with open('/mnt/i/Github/Latent_Style/SchrodingerBridge/configs/630_latent_256_photo2art.json') as f:
    c = json.load(f)
t = c.get('training', {})
f = c.get('full_eval', {})
print('full_eval_each_epoch:', t.get('full_eval_each_epoch'))
print('full_eval_defer_until_training_end:', t.get('full_eval_defer_until_training_end'))
print('full_eval_force_regen:', t.get('full_eval_force_regen'))
print('num_epochs:', t.get('num_epochs'))
print('save_interval:', t.get('save_interval'))
print('full_eval.batch_size:', f.get('batch_size'))
print('full_eval.max_src_samples:', f.get('max_src_samples'))
"
echo "===EXISTING CKPTs==="
ls -la /mnt/i/Github/Latent_Style/SchrodingerBridge/exp/latent256_photo2art/latent256_b16_e10/ 2>/dev/null || echo "NONE"
echo "===DEBUG LOG LAST 20==="
tail -20 /mnt/i/exp_256_photo2art/_train_debug.log 2>/dev/null
