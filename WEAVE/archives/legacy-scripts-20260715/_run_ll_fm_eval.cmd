@echo off
cd /d I:\Github\Latent_Style\SchrodingerBridge
set PYTHONPATH=.
set LOG=C:\Users\Administrator\logs\abl_ll_fm_resume2.log
echo === START %date% %time% === > %LOG%
python -u src\utils\run_evaluation.py --checkpoint exp\abl_no_ll_fm\epoch_0015.pt --output exp\abl_no_ll_fm\full_eval\epoch_0015 --test_dir "I:\datasets\wikiart_distinct5_samam_512_classview\test" --cache_dir "I:\Github\Latent_Style\SchrodingerBridge\exp\eval_cache" --clip_hf_cache_dir "C:\Users\Administrator\.cache\huggingface\hub" --batch_size 2 --generation_batch_size 2 --metric_batch_size 2 --target_chunk_size 1 --vae_decode_batch_size 16 --eval_only_lpips_clip_style --eval_lpips_chunk_size 4 >> %LOG% 2>&1
echo === EVAL_DONE exit=%errorlevel% %date% %time% === >> %LOG%
python _compute_dino.py --images_dir "exp\abl_no_ll_fm\full_eval\epoch_0015\images" --test_dir "I:\datasets\wikiart_distinct5_samam_512_classview\test" --dataset wikiart --output "exp\_dino_results\abl_no_ll_fm.json" --hf_cache "C:\Users\Administrator\.cache\huggingface\hub" --max_refs 30 >> %LOG% 2>&1
echo === DINO_DONE exit=%errorlevel% %date% %time% === >> %LOG%
echo === ALL_COMPLETE %date% %time% === >> %LOG%
