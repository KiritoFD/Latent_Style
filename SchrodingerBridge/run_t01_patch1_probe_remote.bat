@echo off
cd /d I:\Github\Latent_Style\SchrodingerBridge
py tools\experiments\run_t01_patch1_probe.py ^
  --base-config exp\diffeomorphic_tangent_sweep\t01_ws0p03_g6_nl0p05\config.json ^
  --config-root configs\t01_patch1_probe ^
  --output-root exp\t01_patch1_probe ^
  --num-epochs 8 ^
  --train-batch-size 128 ^
  --eval-epochs 6,7,8 ^
  --eval-num-steps 12 ^
  --eval-step-size 1.0 ^
  --eval-vae-decode-scale 0.197 ^
  --eval-residual-scale 1.0
