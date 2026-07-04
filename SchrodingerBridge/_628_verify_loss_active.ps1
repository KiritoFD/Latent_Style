# Run single extreme experiment (X1_velmag_w10) to verify loss is REAL (non-zero)
$root = 'I:/Github/Latent_Style/SchrodingerBridge'
$python = 'C:\Progra~1\Python312\python.exe'
$runScript = "$root\src\run.py"
$configPath = "$root\configs\ablations\628_destructive\X1_velmag_w10.json"
$logPath = "$root\exp\628_ablation\destructive_logs\X1_velmag_w10_verify.log"

Write-Host "=== Verifying loss is REAL (non-zero) with X1_velmag_w10 ==="
Write-Host "Config: $configPath"
Write-Host "Log: $logPath"

# Run for a short time (we just need to see the first few training steps)
# Use timeout via Start-Process with -PassThru and kill after 90s
$proc = Start-Process -FilePath $python `
    -ArgumentList "`"$runScript`" --config `"$configPath`"" `
    -WorkingDirectory $root `
    -RedirectStandardOutput $logPath `
    -RedirectStandardError "$logPath.err" `
    -WindowStyle Hidden `
    -PassThru

Write-Host "Started PID=$($proc.Id), waiting 90s for first training steps..."
Start-Sleep -Seconds 90

if (-not $proc.HasExited) {
    Write-Host "Process still running, killing it (we just needed to verify loss values)..."
    Stop-Process -Id $proc.Id -Force -ErrorAction SilentlyContinue
    Start-Sleep -Seconds 2
}

# Check the log for loss values
Write-Host "`n=== Training log (first 80 lines) ==="
if (Test-Path $logPath) {
    Get-Content $logPath -Head 80
} else {
    Write-Host "Log file not found"
}

Write-Host "`n=== Searching for loss_velocity_magnitude in log ==="
if (Test-Path $logPath) {
    $matches = Select-String -Path $logPath -Pattern 'loss_velocity_magnitude|vel_mag|velocity_ratio|v_pred_norm' -AllMatches
    if ($matches) {
        foreach ($m in $matches | Select-Object -First 10) {
            Write-Host "  $($m.Line)"
        }
    } else {
        Write-Host "  (no matches found - loss may not be logged)"
    }
}

Write-Host "`n=== Searching for loss_directional_cosine (should be 0 since not enabled) ==="
if (Test-Path $logPath) {
    $matches = Select-String -Path $logPath -Pattern 'loss_directional_cosine|dir_cosine' -AllMatches
    if ($matches) {
        foreach ($m in $matches | Select-Object -First 5) {
            Write-Host "  $($m.Line)"
        }
    }
}

# Check for errors
Write-Host "`n=== Error log tail ==="
$errLog = "$logPath.err"
if (Test-Path $errLog) {
    $errSize = (Get-Item $errLog).Length
    if ($errSize -gt 0) {
        Get-Content $errLog -Tail 20
    } else {
        Write-Host "(empty - no errors)"
    }
}
