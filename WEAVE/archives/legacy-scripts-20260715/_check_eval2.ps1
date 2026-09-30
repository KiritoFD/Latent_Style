$base = "I:\Github\Latent_Style\SchrodingerBridge\exp\refactor_verify"
if (Test-Path $base) {
    Write-Output "=== refactor_verify dir exists ==="
    Get-ChildItem $base -Recurse | Select-Object FullName, Length
} else {
    Write-Output "=== refactor_verify dir NOT FOUND ==="
}

# Check python
$py = Get-Process python -ErrorAction SilentlyContinue
if ($py) {
    Write-Output "Python still running: PID=$($py.Id)"
} else {
    Write-Output "Python finished"
}

# Check log file
$log = "C:\Users\Administrator\logs\refactor_verify_eval.log"
if (Test-Path $log) {
    Write-Output "=== LOG (tail 20) ==="
    Get-Content $log -Tail 20
} else {
    Write-Output "=== LOG NOT FOUND ==="
}
