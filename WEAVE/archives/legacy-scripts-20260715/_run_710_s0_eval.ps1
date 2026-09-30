Set-Location "I:\Github\Latent_Style\SchrodingerBridge"
$env:PYTHONIOENCODING = "utf-8"

$ckpt = "exp\710_b0_weave\epoch_0010.pt"
$evalDir = "exp\710_b0_weave\full_eval\epoch_0010"
$testDir = "I:\datasets\wikiart_distinct5_samam_512_classview\test"
$cacheDir = "I:\Github\Latent_Style\SchrodingerBridge\exp\eval_cache"
$hfCache = "I:\Github\Latent_Style\SchrodingerBridge\exp\eval_cache\hf"

# Clean stale eval dir
if (Test-Path $evalDir) { Remove-Item -Recurse -Force $evalDir }

# Run eval (CLIP-S + LPIPS)
$evalStart = Get-Date
python -u src\utils\run_evaluation.py `
    --checkpoint $ckpt `
    --output $evalDir `
    --test_dir $testDir `
    --cache_dir $cacheDir `
    --clip_hf_cache_dir $hfCache `
    --batch_size 2 --generation_batch_size 2 --metric_batch_size 2 `
    --target_chunk_size 5 --vae_decode_batch_size 8 `
    --eval_only_lpips_clip_style --eval_lpips_chunk_size 4 --clip_allow_network `
    --save_generated_images --no-save_summary_grid `
    *> exp\710_b0_weave_eval_log.txt 2>&1
$evalMin = [math]::Round(((Get-Date) - $evalStart).TotalMinutes, 1)
Write-Host "EVAL: ${evalMin}min"

# Run canonical DINO metrics
$dinoStart = Get-Date
python -u src\utils\compute_dino_metrics.py `
    --eval_dir $evalDir `
    --test_dir $testDir `
    --batch_size 4 --max_refs_per_style 30 `
    --exclude_source_from_style_refs `
    --allow_network `
    *> exp\710_b0_weave_dino_log.txt 2>&1
$dinoMin = [math]::Round(((Get-Date) - $dinoStart).TotalMinutes, 1)
Write-Host "DINO: ${dinoMin}min"

# Extract metrics
python -c "import csv,json; rows=list(csv.DictReader(open(r'$evalDir\metrics.csv',encoding='utf-8'))); n=len(rows); off=[i for i,r in enumerate(rows) if r['src_style']!=r['tgt_style']]; no=len(off); ac=sum(float(r['clip_style']) for r in rows)/n; al=sum(float(r['content_lpips']) for r in rows)/n; oc=sum(float(rows[i]['clip_style']) for i in off)/no; ol=sum(float(rows[i]['content_lpips']) for i in off)/no; s=json.load(open(r'$evalDir\dino_summary.json',encoding='utf-8')); print(f'S0_WEAVE n={n} n_off={no}'); print(f'all_clip_s={ac:.4f} all_lpips={al:.4f} all_dino_s={s[\"all_dino_s\"]:.4f} all_dino_c={s[\"all_dino_c\"]:.4f} all_dino_struct={s[\"all_dino_structure\"]:.6f}'); print(f'off_clip_s={oc:.4f} off_lpips={ol:.4f} off_dino_s={s[\"off_dino_s\"]:.4f} off_dino_c={s[\"off_dino_c\"]:.4f} off_dino_struct={s[\"off_dino_structure\"]:.6f}')"
