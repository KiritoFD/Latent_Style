$ErrorActionPreference = 'Continue'

Write-Host "=== List existing schtasks ==="
$ssh_out = ssh -p 2222 -o LogLevel=ERROR administrator@100.115.18.62 "schtasks /Query /FO CSV /NH | findstr /I stylealign samam"
Write-Host "Existing tasks: $ssh_out"

Write-Host ""
Write-Host "=== Try deleting and recreating ==="
$ssh_out = ssh -p 2222 -o LogLevel=ERROR administrator@100.115.18.62 "schtasks /Delete /TN stylealigned_inference /F 2>&1"
Write-Host "Delete: $ssh_out"

Write-Host ""
Write-Host "=== Recreate ==="
$ssh_out = ssh -p 2222 -o LogLevel=ERROR administrator@100.115.18.62 "schtasks /Create /TN stylealigned_inference /TR `"cmd /c C:\Users\Administrator\miniconda3\python.exe C:\Users\Administrator\_run_stylealigned_remote.py > C:\Users\Administrator\logs\stylealigned_run.log 2>&1`" /SC ONCE /ST 00:00 /F"
Write-Host "Create: $ssh_out"

Write-Host ""
Write-Host "=== Run ==="
$ssh_out = ssh -p 2222 -o LogLevel=ERROR administrator@100.115.18.62 "schtasks /Run /TN stylealigned_inference"
Write-Host "Run: $ssh_out"
