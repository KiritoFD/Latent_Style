@echo off
cd /d I:\Github\Latent_Style
py SchrodingerBridge\tools\experiments\build_offline_dino_pairing_plan.py ^
  --cache eval_cache/offline_pairing/dinov2_small_train_cache.pt ^
  --output eval_cache/offline_pairing/dinov2_small_train_pairing_top8.pt ^
  --topk 8 ^
  1> eval_cache\offline_pairing\dinov2_pairing_plan_stdout.log ^
  2> eval_cache\offline_pairing\dinov2_pairing_plan_stderr.log
