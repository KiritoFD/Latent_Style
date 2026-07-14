# Remote: Evaluate StyleID P2A-256 images (CLIP-S + LPIPS)
$ErrorActionPreference = "Continue"
Set-Location "I:\Github\Latent_Style\SchrodingerBridge"

$logDir = "C:\Users\Administrator\logs"
if (-not (Test-Path $logDir)) { New-Item -ItemType Directory -Force -Path $logDir | Out-Null }
$logFile = "$logDir\eval_styleid_p2a_$(Get-Date -Format 'yyyyMMdd_HHmmss').log"
Start-Transcript -Path $logFile -Force

Write-Output "=== StyleID P2A-256 Evaluation ==="
Write-Output "Started: $(Get-Date)"

python tools/eval_clip_lpips_other5.py `
    --gen-dir "I:\exp_256_photo2art\styleid_256\images" `
    --test-dir "I:\datasets\legacy256_overfit50\test" `
    --output-dir "I:\exp_256_photo2art\styleid_256\eval" `
    --style-names "cezanne,Hayao,monet,photo,vangogh" `
    --num-src 30 `
    --clip-local-dir "nonexistent" `
    --batch-size 8

Write-Output "Finished: $(Get-Date)"
Write-Output "Exit code: $LASTEXITCODE"
Stop-Transcript
