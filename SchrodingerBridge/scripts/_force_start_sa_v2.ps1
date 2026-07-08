$ErrorActionPreference = 'Continue'

Write-Host "=== Delete locked log file and recreate ==="
$ssh_out = ssh -p 2222 -o LogLevel=ERROR administrator@100.115.18.62 "del /F C:\Users\Administrator\logs\stylealigned_run.log 2>&1 & echo DONE"
Write-Host "Delete: $ssh_out"

Write-Host ""
Write-Host "=== Recreate empty log ==="
$ssh_out = ssh -p 2222 -o LogLevel=ERROR administrator@100.115.18.62 "type nul > C:\Users\Administrator\logs\stylealigned_run.log 2>&1 & echo DONE"
Write-Host "Recreate: $ssh_out"

Start-Sleep -Seconds 2

Write-Host ""
Write-Host "=== Start StyleAligned ==="
$ssh_out = ssh -p 2222 -o LogLevel=ERROR administrator@100.115.18.62 "cmd /c `"start /B C:\Users\Administrator\miniconda3\python.exe C:\Users\Administrator\_run_stylealigned_remote.py > C:\Users\Administrator\logs\stylealigned_run.log 2>&1`""
Write-Host "Start: exit=$LASTEXITCODE"

Start-Sleep -Seconds 25

Write-Host ""
Write-Host "=== Check log ==="
$ssh_out = ssh -p 2222 -o LogLevel=ERROR administrator@100.115.18.62 "powershell -NoProfile -Command Get-Content C:\Users\Administrator\logs\stylealigned_run.log -Tail 25 -ErrorAction SilentlyContinue"
Write-Host $ssh_out

Write-Host ""
Write-Host "=== GPU ==="
$ssh_out = ssh -p 2222 -o LogLevel=ERROR administrator@100.115.18.62 "nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv,noheader"
Write-Host $ssh_out
