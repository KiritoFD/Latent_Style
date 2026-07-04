# Capture N1 training stderr using Start-Process (avoids fortrl console detection)
$base = "I:\Github\Latent_Style\SchrodingerBridge"
Set-Location $base
$env:PYTHONPATH = ""
# Set FOR_DISABLE_CONSOLE to suppress fortrl window-CLOSE abort
$env:FOR_DISABLE_CONSOLE = "1"

$errLog = "$base\exp\p4_fusion_breakout\n1_lvl2_stderr.log"
$outLog = "$base\exp\p4_fusion_breakout\n1_lvl2_stdout.log"

Write-Host "Starting N1 training with stderr/stdout capture via Start-Process..."
Write-Host "stderr -> $errLog"
Write-Host "stdout -> $outLog"

# Start-Process with redirection does NOT create a console window for the child,
# so fortrl runtime should not detect window-CLOSE event.
$proc = Start-Process -FilePath "C:\Program Files\Python312\python.exe" `
    -ArgumentList "run.py --config configs/p4_n1_lvl2.json" `
    -WorkingDirectory $base `
    -RedirectStandardError $errLog `
    -RedirectStandardOutput $outLog `
    -WindowStyle Hidden `
    -PassThru

Write-Host "Started PID=$($proc.Id), waiting 120s to capture initial output..."
Start-Sleep -Seconds 120

# Check if process is still running
$refreshed = Get-Process -Id $proc.Id -ErrorAction SilentlyContinue
if ($refreshed) {
    Write-Host "[OK] Process still running after 120s, PID=$($proc.Id) CPU=$($refreshed.CPU)s Mem=$([math]::Round($refreshed.WorkingSet64/1MB,1))MB"
    Write-Host "Training appears to be running successfully!"
} else {
    Write-Host "[FAIL] Process exited within 120s"
    Write-Host "Exit code: $($proc.ExitCode)"
}

Write-Host ""
Write-Host "=== stderr log (last 50 lines) ==="
if (Test-Path $errLog) {
    $size = (Get-Item $errLog).Length
    Write-Host "stderr size = $size bytes"
    if ($size -gt 0) {
        Get-Content $errLog -Tail 50
    }
} else {
    Write-Host "stderr log not created"
}

Write-Host ""
Write-Host "=== stdout log (last 30 lines) ==="
if (Test-Path $outLog) {
    $size = (Get-Item $outLog).Length
    Write-Host "stdout size = $size bytes"
    if ($size -gt 0) {
        Get-Content $outLog -Tail 30
    }
} else {
    Write-Host "stdout log not created"
}
