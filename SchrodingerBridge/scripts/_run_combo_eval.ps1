Set-Location "I:\Github\Latent_Style\SchrodingerBridge"
$env:PYTHONIOENCODING = "utf-8"
python -u src\utils\run_evaluation.py `
    --checkpoint exp\evo_d5_combo\epoch_0005.pt `
    --output exp\evo_d5_combo\full_eval\epoch_0005 `
    --test_dir I:\datasets\wikiart_distinct5_samam_512_classview\test `
    --cache_dir exp\eval_cache `
    --clip_hf_cache_dir exp\eval_cache\hf `
    --batch_size 2 --generation_batch_size 2 --metric_batch_size 2 `
    --target_chunk_size 5 --vae_decode_batch_size 8 `
    --eval_only_lpips_clip_style --eval_lpips_chunk_size 4 --clip_allow_network `
    *> exp\evo_d5_combo_eval2_log.txt
