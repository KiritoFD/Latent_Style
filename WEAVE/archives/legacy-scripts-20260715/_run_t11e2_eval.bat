@echo off
cd /d I:\Github\Latent_Style\SchrodingerBridge
set PYTHONPATH=.
set PYTHONIOENCODING=utf-8
python -u src\utils\run_evaluation.py --checkpoint exp\t11e2_extrap05_15ep\epoch_0015.pt --output exp\t11e2_extrap05_15ep\full_eval\epoch_0015 --test_dir I:\datasets\wikiart_distinct5_samam_512_classview\test --cache_dir I:\Github\Latent_Style\SchrodingerBridge\exp\eval_cache --clip_hf_cache_dir C:\Users\Administrator\.cache\huggingface\hub --batch_size 2 --generation_batch_size 2 --metric_batch_size 2 --target_chunk_size 1 --vae_decode_batch_size 16 --eval_only_lpips_clip_style --eval_lpips_chunk_size 4 > C:\Users\Administrator\logs\t11e2_fulleval.out 2> C:\Users\Administrator\logs\t11e2_fulleval.err
