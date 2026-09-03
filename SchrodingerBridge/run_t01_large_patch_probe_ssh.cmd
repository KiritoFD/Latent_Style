@echo off
setlocal
cd /d I:\Github\Latent_Style\SchrodingerBridge
if not exist logs mkdir logs
python tools\experiments\run_t01_large_patch_probe.py ^
  --base-config exp\diffeomorphic_tangent_sweep\t01_ws0p03_g6_nl0p05\config.json ^
  --config-root configs\t01_large_patch_probe ^
  --output-root exp\t01_large_patch_probe ^
  --num-epochs 8 ^
  --eval-epochs 1,4,6,8 ^
  --max-total 3 ^
  --eval-num-steps 12 ^
  --eval-step-size 1.0 ^
  --eval-vae-decode-scale 0.191 ^
  --eval-residual-scale 1.0 ^
  1> logs\t01_large_patch_probe_ssh.out.log 2> logs\t01_large_patch_probe_ssh.err.log
endlocal
