@echo off
setlocal
cd /d I:\Github\Latent_Style\SchrodingerBridge
py -3 -B tools\experiments\run_vae_backend_256_probe.py --variants ema_bodyregion_w34_guard,ema_bodyregion_w42_style --epochs 1 --eval-epochs none --max-train-batches 30 --skip-existing-latents --out-root exp\vae_backend\ema_bodyregion_smoke
endlocal
