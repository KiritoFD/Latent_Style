# Stage7: 从远程拉取 checkpoint + 本地 eval 加速 (RTX 4070 8GB)
# 1. scp checkpoint 从远程 I: 到本地 G:
# 2. 本地跑 CLIP-S/LPIPS eval (batch_size=2, 适配 8GB VRAM)
# 3. 本地跑 DINO eval
$ErrorActionPreference = "Continue"
Set-Location "G:\GitHub\Latent_Style\SchrodingerBridge"
$env:PYTHONIOENCODING = "utf-8"

$sshHost = "administrator@100.115.18.62"
$sshPort = "2222"
$sshOpts = @("-o", "LogLevel=ERROR", "-o", "ConnectTimeout=10")
$remoteRoot = "I:/Github/Latent_Style/SchrodingerBridge"

$expName = "stage7_delta"
$localCkptDir = "G:\GitHub\Latent_Style\SchrodingerBridge\exp\sty_inject\$expName"
$remoteCkptDir = "$remoteRoot/exp/sty_inject/$expName"

# Eval paths (local G: drive)
$testDir = "G:/GitHub/Latent_Style/Dataset/distinct5_512/test"
$cacheDir = "G:/GitHub/Latent_Style/SchrodingerBridge/exp/eval_cache"
$hfCache = "G:/GitHub/Latent_Style/SchrodingerBridge/exp/eval_cache/hf"
$styles = "Early_Renaissance,Impressionism,Minimalism,Rococo,Ukiyo_e"

Write-Output "=== STAGE7 FETCH + LOCAL EVAL START $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') ==="

# ===== PHASE 1: FETCH CHECKPOINT FROM REMOTE =====
Write-Output "--- Fetching checkpoint from remote ---"
if (-not (Test-Path $localCkptDir)) {
    New-Item -ItemType Directory -Path $localCkptDir -Force | Out-Null
}

# Check if checkpoint already exists locally
$localCkpt = "$localCkptDir\epoch_0005.pt"
if (-not (Test-Path $localCkpt)) {
    Write-Output "  scp checkpoint: $remoteCkptDir/epoch_0005.pt -> $localCkpt"
    & scp -P $sshPort @sshOpts "${sshHost}:$remoteCkptDir/epoch_0005.pt" $localCkpt
    if ($LASTEXITCODE -ne 0) {
        Write-Output "  ERROR: scp checkpoint failed. Is training done?"
        Write-Output "  Check remote: ssh -p $sshPort $sshHost 'Test-Path $remoteCkptDir\epoch_0005.pt'"
        exit 1
    }
    Write-Output "  Checkpoint fetched successfully."
} else {
    Write-Output "  Checkpoint already exists locally, skipping scp."
}

# ===== PHASE 2: LOCAL CLIP-S + LPIPS EVAL =====
$evalDir = "G:\GitHub\Latent_Style\SchrodingerBridge\exp\sty_inject\${expName}_d5_eval"
$evalLog = "G:\GitHub\Latent_Style\SchrodingerBridge\exp\sty_inject\${expName}_d5_eval.log"
$summaryPath = "$evalDir\full_eval\epoch_0005\summary.json"

Write-Output "--- Local CLIP-S + LPIPS eval START $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') ---"

if (-not (Test-Path $summaryPath)) {
    $evalArgs = @(
        "-u", "src\utils\run_evaluation.py",
        "--checkpoint", $localCkpt,
        "--output", "$evalDir\full_eval\epoch_0005",
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
    Write-Output "  summary.json exists, skipping CLIP/LPIPS eval."
}

# ===== PHASE 3: LOCAL DINO EVAL =====
$dinoOut = "G:\GitHub\Latent_Style\SchrodingerBridge\exp\sty_inject\_dino\${expName}_d5.json"
Write-Output "--- Local DINO eval START $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') ---"

if (-not (Test-Path $dinoOut)) {
    $dinoArgs = @(
        "_compute_dino.py",
        "--images_dir", "$evalDir\full_eval\epoch_0005\images",
        "--test_dir", $testDir,
        "--dataset", "wikiart",
        "--output", $dinoOut,
        "--hf_cache", $hfCache,
        "--max_refs", "30"
    )
    & python @dinoArgs 2>&1 | Tee-Object -FilePath $evalLog -Append
} else {
    Write-Output "  DINO result exists, skipping."
}

# ===== PHASE 4: PRINT RESULTS =====
Write-Output ""
Write-Output "============================================================"
Write-Output "STAGE7 DELTA RESULTS $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')"
Write-Output "============================================================"
if (Test-Path $summaryPath) {
    $summary = Get-Content $summaryPath -Raw | ConvertFrom-Json
    Write-Output "CLIP-S: $($summary.clip_style)"
    Write-Output "LPIPS:  $($summary.lpips)"
}
if (Test-Path $dinoOut) {
    $dino = Get-Content $dinoOut -Raw | ConvertFrom-Json
    Write-Output "DINO-C: $($dino.dino_con)"
    Write-Output "DINO-S: $($dino.dino_sty)"
}
Write-Output "============================================================"
