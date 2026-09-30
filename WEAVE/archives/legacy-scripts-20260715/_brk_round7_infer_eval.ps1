# Round 7 inference-time experiment: test zero_step_wct / progressive_alpha on brk_a checkpoint
param(
    [Parameter(Mandatory=$true)][string]$OverrideName,  # e.g. _brk_round7_zwct_a03
    [string]$BaseExpName = "brk_a_ll03_10ep",
    [int]$Epoch = 10
)
$ErrorActionPreference = "Continue"
Set-Location "G:\GitHub\Latent_Style\SchrodingerBridge"
$env:PYTHONIOENCODING = "utf-8"

$epochStr = "{0:D4}" -f $Epoch
$localCkpt = "G:\GitHub\Latent_Style\SchrodingerBridge\exp\dino_s_break\$BaseExpName\epoch_${epochStr}.pt"
$overrideFile = "G:\GitHub\Latent_Style\SchrodingerBridge\configs\$OverrideName.json"
$evalDir = "G:\GitHub\Latent_Style\SchrodingerBridge\exp\dino_s_break\${BaseExpName}_${OverrideName}_eval"
$evalLog = "G:\GitHub\Latent_Style\SchrodingerBridge\exp\dino_s_break\${BaseExpName}_${OverrideName}_eval.log"
$summaryPath = "$evalDir\full_eval\epoch_${epochStr}\summary.json"

$testDir = "G:/GitHub/Latent_Style/Dataset/distinct5_512/test"
$cacheDir = "G:/GitHub/Latent_Style/SchrodingerBridge/exp/eval_cache"
$hfCache = "G:/GitHub/Latent_Style/SchrodingerBridge/exp/eval_cache/hf"
$styles = "Early_Renaissance,Impressionism,Minimalism,Rococo,Ukiyo_e"

Write-Output "=== ROUND7 INFER EVAL $OverrideName on $BaseExpName START $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') ==="

# PHASE 1: CLIP-S + LPIPS with config_override
if (-not (Test-Path $summaryPath)) {
    Write-Output "--- CLIP-S + LPIPS eval with override $OverrideName ---"
    $evalArgs = @(
        "-u", "src\utils\run_evaluation.py",
        "--checkpoint", $localCkpt,
        "--output", "$evalDir\full_eval\epoch_${epochStr}",
        "--test_dir", $testDir,
        "--style_subdirs", $styles,
        "--cache_dir", $cacheDir,
        "--clip_hf_cache_dir", $hfCache,
        "--config_override", $overrideFile,
        "--batch_size", "2", "--generation_batch_size", "2", "--metric_batch_size", "2",
        "--target_chunk_size", "1", "--vae_decode_batch_size", "8",
        "--eval_only_lpips_clip_style", "--eval_lpips_chunk_size", "4"
    )
    & python @evalArgs 2>&1 | Tee-Object -FilePath $evalLog
} else {
    Write-Output "--- CLIP/LPIPS summary exists, skipping ---"
}

# PHASE 2: DINO
$dinoOut = "G:\GitHub\Latent_Style\SchrodingerBridge\exp\dino_s_break\_dino\${BaseExpName}_${OverrideName}.json"
if (-not (Test-Path $dinoOut)) {
    Write-Output "--- DINO eval ---"
    $dinoOutDir = Split-Path $dinoOut -Parent
    if (-not (Test-Path $dinoOutDir)) {
        New-Item -ItemType Directory -Path $dinoOutDir -Force | Out-Null
    }
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
} else {
    Write-Output "--- DINO result exists, skipping ---"
}

# PHASE 3: PRINT RESULTS
Write-Output ""
Write-Output "============================================================"
Write-Output "ROUND7 $OverrideName on $BaseExpName RESULTS $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')"
Write-Output "============================================================"
if (Test-Path $summaryPath) {
    $summary = Get-Content $summaryPath -Raw | ConvertFrom-Json
    $o = $summary.analysis.all_pairs_overview
    Write-Output "CLIP-S: $($o.clip_style)"
    Write-Output "LPIPS:  $($o.content_lpips)"
}
if (Test-Path $dinoOut) {
    $dino = Get-Content $dinoOut -Raw | ConvertFrom-Json
    Write-Output "DINO-C: $($dino.dino_content)"
    Write-Output "DINO-S: $($dino.dino_style)"
}
Write-Output "============================================================"
