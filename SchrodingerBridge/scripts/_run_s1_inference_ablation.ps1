Set-Location "I:\Github\Latent_Style\SchrodingerBridge"
$env:PYTHONIOENCODING = "utf-8"

# S0 WEAVE checkpoint
$ckpt = "exp\710_b0_weave\epoch_0010.pt"
$testDir = "I:\datasets\wikiart_distinct5_samam_512_classview\test"
$cacheDir = "I:\Github\Latent_Style\SchrodingerBridge\exp\eval_cache"
$hfCacheDir = "I:\Github\Latent_Style\SchrodingerBridge\exp\eval_cache\hf"

# Override configs to test
$overrides = @(
    @("s1_per_sub_base",     "configs\override_per_subband_base.json"),
    @("s1_per_sub_hh115",    "configs\override_per_subband_hh115.json"),
    @("s1_per_sub_hh125",    "configs\override_per_subband_hh125_lh105.json"),
    @("s1_extrap02",         "configs\override_extrap02.json"),
    @("s1_extrap03",         "configs\override_extrap03.json")
)

$resultsFile = "exp\s1_inference_ablation_results.txt"
"run,n_all,n_off,all_clip_s,all_lpips,all_dino_s,all_dino_c,all_dino_structure,off_clip_s,off_lpips,off_dino_s,off_dino_c,off_dino_structure" | Out-File $resultsFile -Encoding utf8

foreach ($ov in $overrides) {
    $name = $ov[0]
    $overridePath = $ov[1]
    $evalDir = "exp\710_b0_weave\full_eval_s1\$name"

    Write-Host "`n========== $name =========="
    Write-Host "Override: $overridePath"
    Write-Host "Output: $evalDir"

    # Run evaluation with config override
    $evalStart = Get-Date
    python -u src\utils\run_evaluation.py `
        --checkpoint $ckpt `
        --output $evalDir `
        --config_override $overridePath `
        --test_dir $testDir `
        --cache_dir $cacheDir `
        --clip_hf_cache_dir $hfCacheDir `
        --num_steps 8 `
        --batch_size 2 `
        --target_chunk_size 1 `
        --vae_decode_batch_size 16 `
        *> "exp\710_b0_weave\full_eval_s1\${name}_eval_log.txt" 2>&1
    $evalMin = [math]::Round(((Get-Date) - $evalStart).TotalMinutes, 1)
    Write-Host "Eval done: ${evalMin}min"

    if (-not (Test-Path "$evalDir\metrics.csv")) {
        Write-Host "ERROR: metrics.csv not found"
        "$name,ERROR,0,0,0,0,0,0,0,0,0,0,0" | Out-File $resultsFile -Encoding utf8 -Append
        continue
    }

    # Run canonical DINO metrics
    $dinoStart = Get-Date
    python -u src\utils\compute_dino_metrics.py `
        --eval_dir $evalDir `
        --test_dir $testDir `
        --batch_size 4 --max_refs_per_style 30 `
        --exclude_source_from_style_refs `
        --allow_network `
        *> "exp\710_b0_weave\full_eval_s1\${name}_dino_log.txt" 2>&1
    $dinoMin = [math]::Round(((Get-Date) - $dinoStart).TotalMinutes, 1)
    Write-Host "DINO done: ${dinoMin}min"

    # Extract metrics
    $summaryPath = "$evalDir\dino_summary.json"
    if (Test-Path $summaryPath) {
        python -u C:\Users\Administrator\_710_extract_run.py $evalDir $name | Out-File $resultsFile -Encoding utf8 -Append
    } else {
        Write-Host "ERROR: dino_summary.json not found"
        "$name,ERROR,0,0,0,0,0,0,0,0,0,0,0" | Out-File $resultsFile -Encoding utf8 -Append
    }
}

Write-Host "`n========== ALL DONE =========="
Get-Content $resultsFile
