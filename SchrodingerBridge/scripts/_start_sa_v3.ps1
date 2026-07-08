$ErrorActionPreference = 'Continue'

Write-Host "=== Start with NEW log filename ==="
$ssh_out = ssh -p 2222 -o LogLevel=ERROR administrator@100.115.18.62 "cmd /c `"start /B C:\Users\Administrator\miniconda3\python.exe C:\Users\Administrator\_run_stylealigned_remote.py > C:\Users\Administrator\logs\sa_run2.log 2>&1`""
Write-Host "Start: exit=$LASTEXITCODE"

Start-Sleep -Seconds 30

Write-Host ""
Write-Host "=== Check sa_run2.log ==="
$ssh_out = ssh -p 2222 -o LogLevel=ERROR administrator@100.115.18.62 "powershell -NoProfile -Command if (Test-Path C:\Users\Administrator\logs\sa_run2.log) { Get-Content C:\Users\Administrator\logs\sa_run2.log } else { Write-Host 'NOT FOUND' }"
Write-Host $ssh_out

Write-Host ""
Write-Host "=== GPU ==="
$ssh_out = ssh -p 2222 -o LogLevel=ERROR administrator@100.115.18.62 "nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv,noheader"
Write-Host $ssh_out
