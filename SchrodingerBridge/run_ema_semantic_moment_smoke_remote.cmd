@echo off
cd /d I:\Github\Latent_Style\SchrodingerBridge
"C:\Program Files\Python312\python.exe" tools\experiments\run_vae_backend_256_probe.py --variants ema_semantic_moment_adain_w30_guard,ema_semantic_moment_adain_w38_style --epochs 1 --eval-epochs none --skip-existing-latents --max-train-batches 30 --out-root exp\vae_backend\ema_semantic_moment_smoke
