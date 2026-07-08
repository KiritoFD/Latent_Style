$ErrorActionPreference = 'Continue'

Write-Host "=== Create schtasks for StyleAligned ==="
$ssh_out = ssh -p 2222 -o LogLevel=ERROR administrator@100.115.18.62 "schtasks /Create /TN stylealigned_inference /TR `"cmd /c C:\Users\Administrator\miniconda3\python.exe C:\Users\Administrator\_run_stylealigned_remote.py > C:\Users\Administrator\logs\stylealigned_run.log 2>&1`" /SC ONCE /ST 00:00 /F"
Write-Host "Create: $ssh_out"

Write-Host ""
Write-Host "=== Run now ==="
$ssh_out = ssh -p 2222 -o LogLevel=ERROR administrator@100.115.18.62 "schtasks /Run /TN stylealigned_inference"
Write-Host "Run: $ssh_out"

Start-Sleep -Seconds 15

Write-Host ""
Write-Host "=== Check log after 15s ==="
$ssh_out = ssh -p 2222 -o LogLevel=ERROR administrator@100.115.18.62 "powershell -NoProfile -Command if (Test-Path C:\Users\Administrator\logs\stylealigned_run.log) { Get-Content C:\Users\Administrator\logs\stylealigned_run.log -Tail 20 } else { Write-Host 'Log not found' }"
Write-Host $ssh_out

Write-Host ""
Write-Host "=== GPU status ==="
$ssh_out = ssh -p 2222 -o LogLevel=ERROR administrator@100.115.18.62 "nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv,noheader"
Write-Host $ssh_out
