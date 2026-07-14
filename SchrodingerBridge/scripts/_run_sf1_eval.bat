@echo off
cd /d I:\Github\Latent_Style\SchrodingerBridge
set PYTHONPATH=src
python run_evaluation.py --checkpoint I:\Github\Latent_Style\SchrodingerBridge\exp\712_sf1_subband\epoch_0005.pt --output I:\Github\Latent_Style\SchrodingerBridge\exp\712_sf1_subband\full_eval --batch_size 2 --ref_feature_batch_size 2 --vae_decode_batch_size 16 --test_dir I:\datasets\wikiart_distinct5_samam_512_classview\test 2>&1
echo EXIT_CODE=%ERRORLEVEL%
