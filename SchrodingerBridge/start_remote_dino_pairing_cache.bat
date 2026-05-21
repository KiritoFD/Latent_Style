@echo off
cd /d I:\Github\Latent_Style
py SchrodingerBridge\tools\experiments\build_offline_dino_pairing_cache.py ^
  --image-root style_data/train ^
  --latent-root latent-256 ^
  --output eval_cache/offline_pairing/dinov2_small_train_cache.pt ^
  --batch-size 24 ^
  --device cuda ^
  1> eval_cache\offline_pairing\dinov2_cache_stdout.log ^
  2> eval_cache\offline_pairing\dinov2_cache_stderr.log
