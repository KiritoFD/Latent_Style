@echo off
cd /d I:\Github\Latent_Style\SchrodingerBridge
set PYTHONUNBUFFERED=1
if not exist exp\style_embedding_distill\ema_bodytransport_lowfree_w34_fulltrain mkdir exp\style_embedding_distill\ema_bodytransport_lowfree_w34_fulltrain
"C:\Program Files\Python312\python.exe" tools\experiments\run_style_embedding_distill.py --checkpoint exp\vae_backend\ema_bodytransport_lowfree\ema_bodytransport_lowfree_w34_guard\epoch_0006.pt --latent-root I:\Github\Latent_Style\latent-256-sd15-ema --out-root exp\style_embedding_distill\ema_bodytransport_lowfree_w34_fulltrain --recipes d00_emb_only_swd_s4_it60,d02_embspatial_swd_tv_grad_s8_it80 --eval-batch-size 16 --vae-model ema > exp\style_embedding_distill\ema_bodytransport_lowfree_w34_fulltrain\run.log 2>&1
