# Stage10: LL partial style injection training launcher
$ErrorActionPreference = "Continue"
Set-Location "I:\Github\Latent_Style\SchrodingerBridge"
$env:PYTHONIOENCODING = "utf-8"

$logFile = "C:\Users\Administrator\logs\stage10_ll_partial_train.out"
$cfg = "configs\exp_sty_stage10_ll_partial.json"

Write-Output "=== STAGE10 LL_PARTIAL TRAIN START $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') ==="
& python -u src\run.py --config $cfg 2>&1 | Tee-Object -FilePath $logFile
Write-Output "=== STAGE10 LL_PARTIAL TRAIN DONE exit=$LASTEXITCODE $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') ==="
