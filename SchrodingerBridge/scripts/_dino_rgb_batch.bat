@echo off
REM Batch DINO evaluation for hp variants (RGB/latent affine experiments)
cd /d I:\Github\Latent_Style\SchrodingerBridge
set PYTHONPATH=.
set TEST_DIR=I:\datasets\wikiart_distinct5_samam_512_classview\test
set HF_CACHE=C:\Users\Administrator\.cache\huggingface\hub
set OUT_DIR=exp\_dino_results
if not exist %OUT_DIR% mkdir %OUT_DIR%

for %%V in (hp_simple_swd12_15ep hp_lat_s10 hp_rgb_s05 hp_rgb_s10) do (
  set IMG_DIR=exp\%%V\full_eval\epoch_0015\images
  set OUT=%OUT_DIR%\%%V.json
  if exist %OUT_DIR%\%%V.json (
    echo SKIP %%V: already done
  ) else (
    echo === DINO eval: %%V ===
    python _compute_dino.py --images_dir exp\%%V\full_eval\epoch_0015\images --test_dir %TEST_DIR% --dataset wikiart --output %OUT_DIR%\%%V.json --hf_cache %HF_CACHE% --max_refs 30
    echo --- done %%V ---
  )
)
echo ALL DONE
