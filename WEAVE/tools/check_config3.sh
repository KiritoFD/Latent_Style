#!/bin/bash
echo "=== intrinsic_v2 config ==="
python3 -c "
import json
d = json.load(open('/mnt/i/Github/Latent_Style/exp/620_spatial_bridge/620_intrinsic_v2/config.json'))
print('training.test_image_dir:', d.get('training',{}).get('test_image_dir','N/A'))
print('training.train_image_dir:', d.get('training',{}).get('train_image_dir','N/A'))
print('data.latent_root:', d.get('data',{}).get('latent_root','N/A'))
print('data.image_root:', d.get('data',{}).get('image_root','N/A'))
print('model keys:', list(d.get('model',{}).keys())[:15])
print('bridge keys:', list(d.get('bridge',{}).keys())[:15])
fe = d.get('full_eval', {})
print('full_eval.save_generated_images:', fe.get('save_generated_images', 'NOT SET'))
print('full_eval keys:', list(fe.keys()))
"
