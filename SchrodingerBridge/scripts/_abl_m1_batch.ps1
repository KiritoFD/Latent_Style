# M1 Ablation batch: train + full_eval + DINO for 4 subtractive experiments
# Runs sequentially on RTX 3060 12GB. Error handling: continue on failure.
$ErrorActionPreference = "Continue"
Set-Location "I:\Github\Latent_Style\SchrodingerBridge"
$logDir = "C:\Users\Administrator\logs"
$testDir = "I:\datasets\wikiart_distinct5_samam_512_classview\test"
$hfCache = "C:\Users\Administrator\.cache\huggingface\hub"
$dinoOut = "exp\_dino_results"
if (-not (Test-Path $dinoOut)) { New-Item -ItemType Directory -Force -Path $dinoOut | Out-Null }

$exps = @("abl_no_edge", "abl_no_terminal", "abl_no_ep_content", "abl_no_ll_fm")
foreach ($exp in $exps) {
    $logFile = Join-Path $logDir "abl_$exp.log"
    Write-Host "=== START $exp $(Get-Date -Format 'HH:mm:ss') ==="
    # Training (run.py auto-runs full_eval after convergence)
    python -u run.py --config "configs\$exp.json" 2>&1 | Tee-Object -FilePath $logFile
    if ($LASTEXITCODE -ne 0) {
        Write-Host "FAILED $exp training (exit $LASTEXITCODE), continuing to next..."
        continue
    }
    Write-Host "=== TRAIN DONE $exp $(Get-Date -Format 'HH:mm:ss') ==="
    # Find best checkpoint's full_eval images dir
    $evalDirs = Get-ChildItem -Path "exp\$exp\full_eval" -Directory -ErrorAction SilentlyContinue | Sort-Object Name -Descending
    if (-not $evalDirs) {
        Write-Host "NO full_eval dir for $exp, skipping DINO..."
        continue
    }
    $evalDir = $evalDirs[0]
    $imgDir = Join-Path $evalDir.FullName "images"
    if (-not (Test-Path $imgDir)) {
        Write-Host "NO images dir for $exp at $imgDir, skipping DINO..."
        continue
    }
    # DINO evaluation
    $dinoJson = Join-Path $dinoOut "$exp.json"
    if (Test-Path $dinoJson) {
        Write-Host "DINO already done for $exp, skipping..."
    } else {
        Write-Host "=== DINO $exp ==="
        $env:PYTHONPATH = "."
        python _compute_dino.py --images_dir $imgDir --test_dir $testDir --dataset wikiart --output $dinoJson --hf_cache $hfCache --max_refs 30 2>&1 | Tee-Object -FilePath (Join-Path $logDir "abl_${exp}_dino.log")
    }
    Write-Host "=== ALL DONE $exp $(Get-Date -Format 'HH:mm:ss') ==="
}
Write-Host "=== M1 BATCH COMPLETE $(Get-Date -Format 'HH:mm:ss') ==="
