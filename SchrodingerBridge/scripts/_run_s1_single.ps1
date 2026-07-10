param(
    [string]$RunName,
    [string]$OverrideConfig,
    [string]$Ckpt = "exp\710_b0_weave\epoch_0010.pt",
    [int]$NumSteps = 8,
    [string]$PostprocessMode = "none",
    [double]$PostprocessStrength = 0.0,
    [string]$LatentPostprocessMode = "none",
    [int]$VggOptSteps = 5,
    [double]$VggOptLr = 0.02,
    [double]$VggStyleWeight = 1.0,
    [double]$VggContentWeight = 10.0,
    [int]$VggImageSize = 256,
    [int]$VggRefLimit = 8
)

Set-Location "I:\Github\Latent_Style\SchrodingerBridge"
$env:PYTHONIOENCODING = "utf-8"
$testDir = "I:\datasets\wikiart_distinct5_samam_512_classview\test"
$cacheDir = "I:\Github\Latent_Style\SchrodingerBridge\exp\eval_cache"
$hfCacheDir = "I:\Github\Latent_Style\SchrodingerBridge\exp\eval_cache\hf"
$ckptDir = Split-Path $Ckpt -Parent
$evalDir = "$ckptDir\full_eval_s1\$RunName"

Write-Host "========== $RunName =========="
Write-Host "Override: $OverrideConfig"
Write-Host "Output: $evalDir"
Write-Host "Postprocess: $PostprocessMode (strength=$PostprocessStrength)"
Write-Host "LatentPostprocess: $LatentPostprocessMode"

# Run evaluation with config override
$evalStart = Get-Date
$cmdArgs = @(
    "-u", "src\utils\run_evaluation.py",
    "--checkpoint", $Ckpt,
    "--output", $evalDir,
    "--config_override", $OverrideConfig,
    "--test_dir", $testDir,
    "--cache_dir", $cacheDir,
    "--clip_hf_cache_dir", $hfCacheDir,
    "--num_steps", $NumSteps,
    "--batch_size", 2,
    "--target_chunk_size", 1,
    "--vae_decode_batch_size", 16
)
if ($PostprocessMode -ne "none" -and $PostprocessStrength -gt 0.0) {
    $cmdArgs += @("--postprocess_mode", $PostprocessMode,
                  "--postprocess_strength", $PostprocessStrength,
                  "--allow_metric_postprocess")
}
if ($LatentPostprocessMode -ne "none") {
    $cmdArgs += @("--latent_postprocess_mode", $LatentPostprocessMode,
                  "--vgg_opt_steps", $VggOptSteps,
                  "--vgg_opt_lr", $VggOptLr,
                  "--vgg_style_weight", $VggStyleWeight,
                  "--vgg_content_weight", $VggContentWeight,
                  "--vgg_image_size", $VggImageSize,
                  "--vgg_ref_limit", $VggRefLimit)
}
python @cmdArgs
$evalMin = [math]::Round(((Get-Date) - $evalStart).TotalMinutes, 1)
Write-Host "Eval done: ${evalMin}min"

if (-not (Test-Path "$evalDir\metrics.csv")) {
    Write-Host "ERROR: metrics.csv not found"
    exit 1
}

# Run canonical DINO metrics
$dinoStart = Get-Date
python -u src\utils\compute_dino_metrics.py `
    --eval_dir $evalDir `
    --test_dir $testDir `
    --batch_size 4 --max_refs_per_style 30 `
    --exclude_source_from_style_refs `
    --allow_network
$dinoMin = [math]::Round(((Get-Date) - $dinoStart).TotalMinutes, 1)
Write-Host "DINO done: ${dinoMin}min"

# Extract metrics
python -u C:\Users\Administrator\_710_extract_run.py $evalDir $RunName
