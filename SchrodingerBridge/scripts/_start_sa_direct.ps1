$ErrorActionPreference = 'Continue'

# Direct approach: run the python script directly via SSH, detach with nohup-like trick
Write-Host "=== Starting StyleAligned directly via SSH ==="

# Use cmd /c start to detach the process
$cmd = 'cmd /c "start /B C:\Users\Administrator\miniconda3\python.exe C:\Users\Administrator\_run_stylealigned_remote.py > C:\Users\Administrator\logs\stylealigned_run.log 2>&1"'

$ssh_out = ssh -p 2222 -o LogLevel=ERROR administrator@100.115.18.62 $cmd
Write-Host "Start result: $ssh_out (exit=$LASTEXITCODE)"

Start-Sleep -Seconds 20

Write-Host ""
Write-Host "=== Check log after 20s ==="
$ssh_out = ssh -p 2222 -o LogLevel=ERROR administrator@100.115.18.62 "powershell -NoProfile -Command if (Test-Path C:\Users\Administrator\logs\stylealigned_run.log) { Get-Content C:\Users\Administrator\logs\stylealigned_run.log } else { Write-Host 'Log not found' }"
Write-Host $ssh_out

Write-Host ""
Write-Host "=== GPU status ==="
$ssh_out = ssh -p 2222 -o LogLevel=ERROR administrator@100.115.18.62 "nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv,noheader"
Write-Host $ssh_out
