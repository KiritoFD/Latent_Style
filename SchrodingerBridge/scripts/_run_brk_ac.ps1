# Round 9 brk_ac training launcher
# Runs on remote RTX 3060, redirects output to log file
$ErrorActionPreference = "Continue"
$base = "I:\Github\Latent_Style\SchrodingerBridge"
Set-Location $base
$logFile = "$base\logs\brk_ac_fft_loss_train.log"
$errFile = "$base\logs\brk_ac_fft_loss_err.log"

# Ensure logs directory exists
if (-not (Test-Path "$base\logs")) {
    New-Item -ItemType Directory -Path "$base\logs" -Force | Out-Null
}

Write-Output "[$(Get-Date)] Starting brk_ac FFT loss training..."
Write-Output "Config: $base\configs\exp_brk_ac_fft_loss.json"
Write-Output "Log: $logFile"

python "$base\src\run.py" --config "$base\configs\exp_brk_ac_fft_loss.json" *>&1 | Tee-Object -FilePath $logFile

$exitCode = $LASTEXITCODE
Write-Output "[$(Get-Date)] Training complete. Exit code: $exitCode" | Out-File -Append -FilePath $logFile -Encoding utf8
Write-Output "TRAIN_DONE_EXITCODE=$exitCode" | Out-File -Append -FilePath $logFile -Encoding utf8
