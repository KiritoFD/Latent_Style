# DINO-S break: inference parameter tuning on existing checkpoint (no retraining)
param(
    [Parameter(Mandatory=$true)][string]$ExpName,
    [Parameter(Mandatory=$true)][string]$OverrideName,
    [int]$Epoch = 10
)
$ErrorActionPreference = "Continue"
Set-Location "G:\GitHub\Latent_Style\SchrodingerBridge"
$env:PYTHONIOENCODING = "utf-8"

$localCkpt = "G:\GitHub\Latent_Style\SchrodingerBridge\exp\dino_s_break\$ExpName\epoch_$($('{0:D4}' -f $Epoch)).pt"
$overrideFile = "G:\GitHub\Latent_Style\SchrodingerBridge\configs\${OverrideName}.json"

$testDir = "G:/GitHub/Latent_Style/Dataset/distinct5_512/test"
$cacheDir = "G:/GitHub/Latent_Style/SchrodingerBridge/exp/eval_cache"
$hfCache = "G:/GitHub/Latent_Style/SchrodingerBridge/exp/eval_cache/hf"
$styles = "Early_Renaissance,Impressionism,Minimalism,Rococo,Ukiyo_e"

$evalDir = "G:\GitHub\Latent_Style\SchrodingerBridge\exp\dino_s_break\${ExpName}_${OverrideName}_eval"
$evalLog = "G:\GitHub\Latent_Style\SchrodingerBridge\exp\dino_s_break\${ExpName}_${OverrideName}_eval.log"
$epochStr = '{0:D4}' -f $Epoch
$summaryPath = "$evalDir\full_eval\epoch_${epochStr}\summary.json"

Write-Output "=== INFERENCE TUNE $ExpName + $OverrideName START $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') ==="

# CLIP-S + LPIPS with override
if (-not (Test-Path $summaryPath)) {
    $evalArgs = @(
        "-u", "src\utils\run_evaluation.py",
        "--checkpoint", $localCkpt,
        "--output", "$evalDir\full_eval\epoch_${epochStr}",
        "--test_dir", $testDir,
        "--style_subdirs", $styles,
        "--cache_dir", $cacheDir,
        "--clip_hf_cache_dir", $hfCache,
        "--batch_size", "2", "--generation_batch_size", "2", "--metric_batch_size", "2",
        "--target_chunk_size", "1", "--vae_decode_batch_size", "8",
        "--eval_only_lpips_clip_style", "--eval_lpips_chunk_size", "4",
        "--config_override", $overrideFile
    )
    & python @evalArgs 2>&1 | Tee-Object -FilePath $evalLog
} else {
    Write-Output "--- CLIP/LPIPS summary exists, skipping ---"
}

# DINO
$dinoOut = "G:\GitHub\Latent_Style\SchrodingerBridge\exp\dino_s_break\_dino\${ExpName}_${OverrideName}_d5.json"
if (-not (Test-Path $dinoOut)) {
    $dinoOutDir = Split-Path $dinoOut -Parent
    if (-not (Test-Path $dinoOutDir)) { New-Item -ItemType Directory -Path $dinoOutDir -Force | Out-Null }
    $dinoArgs = @(
        "_compute_dino.py",
        "--images_dir", "$evalDir\full_eval\epoch_${epochStr}\images",
        "--test_dir", $testDir,
        "--dataset", "wikiart",
        "--output", $dinoOut,
        "--hf_cache", $hfCache,
        "--max_refs", "30"
    )
    & python @dinoArgs 2>&1 | Tee-Object -FilePath $evalLog -Append
}

Write-Output "============================================================"
Write-Output "INFERENCE TUNE $ExpName + $OverrideName RESULTS $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')"
Write-Output "============================================================"
if (Test-Path $summaryPath) {
    $summary = Get-Content $summaryPath -Raw | ConvertFrom-Json
    Write-Output "CLIP-S: $($summary.analysis.all_pairs_overview.clip_style)"
    Write-Output "LPIPS:  $($summary.analysis.all_pairs_overview.content_lpips)"
}
if (Test-Path $dinoOut) {
    $dino = Get-Content $dinoOut -Raw | ConvertFrom-Json
    Write-Output "DINO-C: $($dino.dino_con)"
    Write-Output "DINO-S: $($dino.dino_sty)"
}
Write-Output "============================================================"
