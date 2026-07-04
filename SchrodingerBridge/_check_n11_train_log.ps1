# Check N11+N16 training log to understand how it succeeded
$base = "I:\Github\Latent_Style\SchrodingerBridge"

Write-Host "=== N11+N16 Training Logs ==="
$n11Dir = "$base\exp\p4_fusion_breakout\n11_n16_gate03_whh25"
$logsDir = "$n11Dir\logs"
if (Test-Path $logsDir) {
    Write-Host "Logs dir: $logsDir"
    $logFiles = Get-ChildItem $logsDir -Filter "*.csv"
    foreach ($f in $logFiles) {
        Write-Host ""
        Write-Host "--- $($f.Name) (first 5 lines) ---"
        Get-Content $f.FullName -Head 5
    }
}

Write-Host ""
Write-Host "=== Check for n11_train.log in exp/p4_fusion_breakout ==="
$n11Log = "$base\exp\p4_fusion_breakout\n11_n16_gate03_whh25_train.log"
if (Test-Path $n11Log) {
    $size = (Get-Item $n11Log).Length
    Write-Host "[OK] N11 log exists, size=$size bytes"
    Write-Host "--- First 30 lines ---"
    Get-Content $n11Log -Head 30
    Write-Host "--- Last 20 lines ---"
    Get-Content $n11Log -Tail 20
} else {
    Write-Host "[WARN] N11 log not found: $n11Log"
    # Search for any n11 logs
    $allLogs = Get-ChildItem "$base\exp\p4_fusion_breakout" -Filter "*n11*" -Recurse -ErrorAction SilentlyContinue
    if ($allLogs) {
        Write-Host "Found n11 files:"
        $allLogs | ForEach-Object { Write-Host "  $($_.FullName) size=$($_.Length)" }
    }
}

Write-Host ""
Write-Host "=== Check for pythonw.exe availability ==="
$pythonw = "C:\Program Files\Python312\pythonw.exe"
if (Test-Path $pythonw) {
    Write-Host "[OK] pythonw.exe exists: $pythonw"
} else {
    Write-Host "[FAIL] pythonw.exe not found"
    # Search for pythonw
    $found = Get-ChildItem "C:\Program Files\Python312" -Filter "pythonw*" -ErrorAction SilentlyContinue
    if ($found) {
        $found | ForEach-Object { Write-Host "  Found: $($_.FullName)" }
    }
}
