param(
    [string]$ConfigName
)

Set-Location "I:\Github\Latent_Style\SchrodingerBridge"
$env:PYTHONIOENCODING = "utf-8"

$configPath = "configs\$ConfigName.json"
Write-Host "========== Phase S2: $ConfigName =========="
Write-Host "Config: $configPath"

# Read config to get save_dir and num_epochs
$config = Get-Content $configPath -Raw | ConvertFrom-Json
$saveDir = $config.checkpoint.save_dir
$numEpochs = $config.training.num_epochs
$runName = $config.ablation.name
if (-not $runName) { $runName = $ConfigName }

# Convert I: path to relative for checkpoint
$ckptRel = $saveDir -replace 'I:\\Github\\Latent_Style\\SchrodingerBridge\\', ''
$ckptPath = "$ckptRel\epoch_$($numEpochs.ToString('D4')).pt"

Write-Host "Save dir: $saveDir"
Write-Host "Num epochs: $numEpochs"
Write-Host "Checkpoint: $ckptPath"

# Phase 1: Training
$trainStart = Get-Date
Write-Host "`n--- Training ---"
python -u run.py --config $configPath
$trainMin = [math]::Round(((Get-Date) - $trainStart).TotalMinutes, 1)
Write-Host "Training done: ${trainMin}min"

if (-not (Test-Path $ckptPath)) {
    Write-Host "ERROR: Checkpoint not found at $ckptPath"
    exit 1
}

# Phase 2: Evaluation
$testDir = "I:\datasets\wikiart_distinct5_samam_512_classview\test"
$cacheDir = "I:\Github\Latent_Style\SchrodingerBridge\exp\eval_cache"
$hfCacheDir = "I:\Github\Latent_Style\SchrodingerBridge\exp\eval_cache\hf"
$evalDir = "$ckptRel\full_eval"

$evalStart = Get-Date
Write-Host "`n--- Evaluation ---"
python -u src\utils\run_evaluation.py `
    --checkpoint $ckptPath `
    --output $evalDir `
    --test_dir $testDir `
    --cache_dir $cacheDir `
    --clip_hf_cache_dir $hfCacheDir `
    --num_steps 8 `
    --batch_size 2 `
    --target_chunk_size 1 `
    --vae_decode_batch_size 16
$evalMin = [math]::Round(((Get-Date) - $evalStart).TotalMinutes, 1)
Write-Host "Eval done: ${evalMin}min"

if (-not (Test-Path "$evalDir\metrics.csv")) {
    Write-Host "ERROR: metrics.csv not found"
    exit 1
}

# Phase 3: DINO metrics
$dinoStart = Get-Date
Write-Host "`n--- DINO Metrics ---"
python -u src\utils\compute_dino_metrics.py `
    --eval_dir $evalDir `
    --test_dir $testDir `
    --batch_size 4 --max_refs_per_style 30 `
    --exclude_source_from_style_refs `
    --allow_network
$dinoMin = [math]::Round(((Get-Date) - $dinoStart).TotalMinutes, 1)
Write-Host "DINO done: ${dinoMin}min"

# Phase 4: Extract metrics
Write-Host "`n--- Results ---"
python -u C:\Users\Administrator\_710_extract_run.py $evalDir $runName

Write-Host "`n========== DONE: $ConfigName =========="
Write-Host "Train: ${trainMin}min, Eval: ${evalMin}min, DINO: ${dinoMin}min"
