@echo off
cd /d I:\Github\Latent_Style\SchrodingerBridge
"C:\Program Files\Python312\python.exe" src\utils\run_evaluation.py ^
  --checkpoint I:\Github\Latent_Style\exp\aaai2027_phase2_smoe_translator_k070_e3_seed42_b12a1\epoch_0008.pt ^
  --output I:\Github\Latent_Style\exp\aaai2027_phase2_smoe_translator_k070_e3_seed42_b12a1\full_eval\bench_after ^
  --test_dir I:\wikiart_distinct5_samam_512_classview\test ^
  --cache_dir I:\Github\Latent_Style\eval_cache ^
  --clip_hf_cache_dir I:\Github\Latent_Style\eval_cache\hf ^
  --profile_timing ^
  --eval_only_lpips_clip_style ^
  --keep_generated_on_device ^
  --source_latent_cache ^
  --max_src_samples 30 ^
  --force_regen ^
  --no-save_generated_images ^
  --no-save_summary_grid ^
  --no-eval_enable_introstyle ^
  --no-eval_enable_art_fid ^
  --no-eval_enable_kid
