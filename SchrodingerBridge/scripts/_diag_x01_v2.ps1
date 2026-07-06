# Diagnose X01 training - check log file and try direct run
$LOG = "I:\Github\Latent_Style\SchrodingerBridge\logs\abl512_v3_X01_euler_train.log"
$REPO = "I:\Github\Latent_Style\SchrodingerBridge"
$SRC = "$REPO\src"
$PYTHON = "C:\Program Files\Python312\python.exe"
$CONFIG = "$REPO\configs\abl512_X01_euler.json"

Write-Host "=== X01 train log content ==="
if (Test-Path $LOG) {
    $size = (Get-Item $LOG).Length
    Write-Host "Log size: $size bytes"
    if ($size -gt 0) {
        Get-Content $LOG -TotalCount 50
    } else {
        Write-Host "Log is EMPTY"
    }
} else {
    Write-Host "Log not found: $LOG"
}

Write-Host ""
Write-Host "=== Try direct python invocation ==="
$env:PYTHONPATH = "$SRC"
$env:CUDA_VISIBLE_DEVICES = "0"
$env:HF_HOME = "$REPO\exp\eval_cache\hf"
Set-Location $SRC
Write-Host "CWD: $(Get-Location)"
Write-Host "PYTHONPATH: $env:PYTHONPATH"
Write-Host "Config: $CONFIG"
Write-Host ""
Write-Host "=== Running: python -m run --config $CONFIG (first 60 sec) ==="
$proc = Start-Process -FilePath $PYTHON -ArgumentList "-m", "run", "--config", $CONFIG -NoNewWindow -PassThru -RedirectStandardOutput "$REPO\logs\_direct_stdout.txt" -RedirectStandardError "$REPO\logs\_direct_stderr.txt"
Write-Host "Started PID: $($proc.Id)"
Start-Sleep -Seconds 30
if (-not $proc.HasExited) {
    Write-Host "Status: RUNNING after 30s"
    Write-Host "Killing test process..."
    $proc | Stop-Process -Force
} else {
    Write-Host "Status: EXITED with code $($proc.ExitCode) within 30s"
}
Write-Host ""
Write-Host "=== Direct stdout (first 30 lines) ==="
if (Test-Path "$REPO\logs\_direct_stdout.txt") {
    Get-Content "$REPO\logs\_direct_stdout.txt" -TotalCount 30
}
Write-Host ""
Write-Host "=== Direct stderr (first 50 lines) ==="
if (Test-Path "$REPO\logs\_direct_stderr.txt") {
    Get-Content "$REPO\logs\_direct_stderr.txt" -TotalCount 50
}
