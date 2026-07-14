# DINO-S break: fetch checkpoint + local eval
param(
    [Parameter(Mandatory=$true)][string]$ExpName,
    [int]$Epoch = 10
)
$ErrorActionPreference = "Continue"
Set-Location "G:\GitHub\Latent_Style\SchrodingerBridge"
$env:PYTHONIOENCODING = "utf-8"

$sshHost = "administrator@100.115.18.62"
$sshPort = "2222"
$sshOpts = @("-o", "LogLevel=ERROR", "-o", "ConnectTimeout=10")
$remoteRoot = "I:/Github/Latent_Style/SchrodingerBridge"

$localCkptDir = "G:\GitHub\Latent_Style\SchrodingerBridge\exp\dino_s_break\$ExpName"
$remoteCkptDir = "$remoteRoot/exp/dino_s_break/$ExpName"
$epochStr = "{0:D4}" -f $Epoch
$localCkpt = "$localCkptDir\epoch_${epochStr}.pt"

$testDir = "G:/GitHub/Latent_Style/Dataset/distinct5_512/test"
$cacheDir = "G:/GitHub/Latent_Style/SchrodingerBridge/exp/eval_cache"
$hfCache = "G:/GitHub/Latent_Style/SchrodingerBridge/exp/eval_cache/hf"
$styles = "Early_Renaissance,Impressionism,Minimalism,Rococo,Ukiyo_e"

$evalDir = "G:\GitHub\Latent_Style\SchrodingerBridge\exp\dino_s_break\${ExpName}_d5_eval"
$evalLog = "G:\GitHub\Latent_Style\SchrodingerBridge\exp\dino_s_break\${ExpName}_d5_eval.log"
$summaryPath = "$evalDir\full_eval\epoch_${epochStr}\summary.json"

Write-Output "=== BRK EVAL $ExpName epoch=$Epoch START $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') ==="

# PHASE 1: FETCH
if (-not (Test-Path $localCkptDir)) {
    New-Item -ItemType Directory -Path $localCkptDir -Force | Out-Null
}
if (-not (Test-Path $localCkpt)) {
    Write-Output "--- Fetching checkpoint ---"
    & scp -P $sshPort @sshOpts "${sshHost}:$remoteCkptDir/epoch_${epochStr}.pt" $localCkpt
    if ($LASTEXITCODE -ne 0) {
        Write-Output "ERROR: scp failed for $ExpName"
        exit 1
    }
} else {
    Write-Output "--- Checkpoint exists locally ---"
}

# PHASE 2: CLIP-S + LPIPS
if (-not (Test-Path $summaryPath)) {
    Write-Output "--- CLIP-S + LPIPS eval ---"
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
        "--eval_only_lpips_clip_style", "--eval_lpips_chunk_size", "4"
    )
    & python @evalArgs 2>&1 | Tee-Object -FilePath $evalLog
} else {
    Write-Output "--- CLIP/LPIPS summary exists, skipping ---"
}

# PHASE 3: DINO
$dinoOut = "G:\GitHub\Latent_Style\SchrodingerBridge\exp\dino_s_break\_dino\${ExpName}_d5.json"
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

# PHASE 4: PRINT RESULTS
Write-Output ""
Write-Output "============================================================"
Write-Output "BRK $ExpName RESULTS $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')"
Write-Output "============================================================"
if (Test-Path $summaryPath) {
    $summary = Get-Content $summaryPath -Raw | ConvertFrom-Json
    Write-Output "CLIP-S: $($summary.clip_style)"
    Write-Output "LPIPS:  $($summary.content_lpips)"
}
if (Test-Path $dinoOut) {
    $dino = Get-Content $dinoOut -Raw | ConvertFrom-Json
    Write-Output "DINO-C: $($dino.dino_con)"
    Write-Output "DINO-S: $($dino.dino_sty)"
}
Write-Output "============================================================"
