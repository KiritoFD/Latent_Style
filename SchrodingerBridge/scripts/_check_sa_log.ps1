$ErrorActionPreference = 'Continue'

Start-Sleep -Seconds 10

Write-Host "=== StyleAligned run log (first 50 lines) ==="
$ssh_out = ssh -p 2222 -o LogLevel=ERROR administrator@100.115.18.62 "powershell -NoProfile -Command if (Test-Path C:\Users\Administrator\logs\stylealigned_run.log) { Get-Content C:\Users\Administrator\logs\stylealigned_run.log -Tail 30 } else { Write-Host 'Log not found' }"
Write-Host $ssh_out

Write-Host ""
Write-Host "=== GPU status ==="
$ssh_out = ssh -p 2222 -o LogLevel=ERROR administrator@100.115.18.62 "nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv,noheader"
Write-Host $ssh_out
