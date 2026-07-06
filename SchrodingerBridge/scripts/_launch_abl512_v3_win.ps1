# Launcher: starts abl512 v3 batch training in background Windows PowerShell process
# This avoids SSH session termination issues
$REPO = "I:\Github\Latent_Style\SchrodingerBridge"
$SCRIPT = "$REPO\scripts\run_abl512_v3.ps1"
$BATCH_LOG = "$REPO\logs\abl512_v3_batch.log"

# Ensure logs dir exists
New-Item -ItemType Directory -Force -Path "$REPO\logs" | Out-Null

# Kill any existing python.exe running ablation (clean slate)
Get-Process python -ErrorAction SilentlyContinue | Where-Object {
    $_.CommandLine -match "abl512"
} | Stop-Process -Force -ErrorAction SilentlyContinue

# Launch batch training in background
$process = Start-Process powershell -ArgumentList "-ExecutionPolicy Bypass -File `"$SCRIPT`"" -WindowStyle Hidden -PassThru

Write-Host "Launched abl512 v3 batch training (Windows PowerShell background)"
Write-Host "  PID: $($process.Id)"
Write-Host "  Script: $SCRIPT"
Write-Host "  Batch log: $BATCH_LOG"
Start-Sleep -Seconds 3

# Verify it's running
if (-not $process.HasExited) {
    Write-Host "  Status: RUNNING"
    # Show first few lines of batch log
    if (Test-Path $BATCH_LOG) {
        Write-Host "  Batch log first 10 lines:"
        Get-Content $BATCH_LOG -Head 10 | ForEach-Object { "    $_" }
    }
} else {
    Write-Host "  Status: FAILED to start (exit code $($process.ExitCode))"
    if (Test-Path $BATCH_LOG) {
        Write-Host "  Batch log last 20 lines:"
        Get-Content $BATCH_LOG -Tail 20 | ForEach-Object { "    $_" }
    }
}
