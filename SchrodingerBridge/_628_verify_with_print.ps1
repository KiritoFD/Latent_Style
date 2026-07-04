# Clean X1 result and re-run with debug print to verify loss is REAL
$root = 'I:/Github/Latent_Style/SchrodingerBridge'
$python = 'C:\Progra~1\Python312\python.exe'
$runScript = "$root\src\run.py"
$configPath = "$root\configs\ablations\628_destructive\X1_velmag_w10.json"
$logPath = "$root\exp\628_ablation\destructive_logs\X1_velmag_w10_verify2.log"
$expDir = "$root\exp\628_ablation\destructive\X1_velmag_w10"

# Clean old X1 result
if (Test-Path $expDir) {
    Write-Host "Cleaning old X1 result..."
    Remove-Item $expDir -Recurse -Force
}

# Run for 60s then kill
Write-Host "=== Running X1_velmag_w10 with debug print ==="
$proc = Start-Process -FilePath $python `
    -ArgumentList "`"$runScript`" --config `"$configPath`"" `
    -WorkingDirectory $root `
    -RedirectStandardOutput $logPath `
    -RedirectStandardError "$logPath.err" `
    -WindowStyle Hidden `
    -PassThru

Write-Host "Started PID=$($proc.Id), waiting 60s..."
Start-Sleep -Seconds 60

if (-not $proc.HasExited) {
    Stop-Process -Id $proc.Id -Force -ErrorAction SilentlyContinue
    Start-Sleep -Seconds 2
}

# Search for 628-DEBUG in log
Write-Host "`n=== Searching for 628-DEBUG in stdout ==="
if (Test-Path $logPath) {
    $matches = Select-String -Path $logPath -Pattern '628-DEBUG' -AllMatches
    if ($matches) {
        foreach ($m in $matches) {
            Write-Host "  $($m.Line)"
        }
    } else {
        Write-Host "  (no 628-DEBUG found - checking full log)"
        Get-Content $logPath -Head 30
    }
}

Write-Host "`n=== Searching for 628-DEBUG in stderr ==="
$errLog = "$logPath.err"
if (Test-Path $errLog) {
    $matches = Select-String -Path $errLog -Pattern '628-DEBUG' -AllMatches
    if ($matches) {
        foreach ($m in $matches) {
            Write-Host "  $($m.Line)"
        }
    }
    # Also check for errors
    $errMatches = Select-String -Path $errLog -Pattern 'Error|Traceback|Exception' -AllMatches
    if ($errMatches) {
        Write-Host "`n=== Errors found ==="
        foreach ($m in $errMatches | Select-Object -First 5) {
            Write-Host "  $($m.Line)"
        }
    }
}

# Clean up the partial result
if (Test-Path $expDir) {
    Remove-Item $expDir -Recurse -Force -ErrorAction SilentlyContinue
}
