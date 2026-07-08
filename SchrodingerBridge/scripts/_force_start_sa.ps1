$ErrorActionPreference = 'Continue'

Write-Host "=== Check processes using python ==="
$ssh_out = ssh -p 2222 -o LogLevel=ERROR administrator@100.115.18.62 "tasklist /FI `"IMAGENAME eq python.exe`" /FO CSV /NH"
Write-Host "Python processes: $ssh_out"

Write-Host ""
Write-Host "=== Check if log file is locked ==="
$ssh_out = ssh -p 2222 -o LogLevel=ERROR administrator@100.115.18.62 "powershell -NoProfile -Command try { [IO.File]::OpenWrite('C:\Users\Administrator\logs\stylealigned_run.log').Close(); Write-Host 'NOT LOCKED' } catch { Write-Host 'LOCKED' }"
Write-Host "Log file: $ssh_out"

Write-Host ""
Write-Host "=== Kill all python processes ==="
$ssh_out = ssh -p 2222 -o LogLevel=ERROR administrator@100.115.18.62 "taskkill /F /IM python.exe 2>&1"
Write-Host "Kill: $ssh_out"

Start-Sleep -Seconds 3

Write-Host ""
Write-Host "=== Try again after kill ==="
$ssh_out = ssh -p 2222 -o LogLevel=ERROR administrator@100.115.18.62 "cmd /c `"start /B C:\Users\Administrator\miniconda3\python.exe C:\Users\Administrator\_run_stylealigned_remote.py > C:\Users\Administrator\logs\stylealigned_run.log 2>&1`""
Write-Host "Start: $ssh_out (exit=$LASTEXITCODE)"

Start-Sleep -Seconds 20

Write-Host ""
Write-Host "=== Check log ==="
$ssh_out = ssh -p 2222 -o LogLevel=ERROR administrator@100.115.18.62 "powershell -NoProfile -Command if (Test-Path C:\Users\Administrator\logs\stylealigned_run.log) { Get-Content C:\Users\Administrator\logs\stylealigned_run.log -Tail 20 } else { Write-Host 'Log not found' }"
Write-Host $ssh_out

Write-Host ""
Write-Host "=== GPU ==="
$ssh_out = ssh -p 2222 -o LogLevel=ERROR administrator@100.115.18.62 "nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv,noheader"
Write-Host $ssh_out
