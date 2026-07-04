$ErrorActionPreference = "SilentlyContinue"
Write-Host "=== PYTHON PROCS ==="
Get-Process python -ErrorAction SilentlyContinue | Select-Object Id, @{N='Mem_MB';E={[math]::Round($_.WorkingSet64/1MB,0)}}, StartTime
Write-Host "=== GPU ==="
nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu --format=csv,noheader
Write-Host "=== GPU PROCS ==="
nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader
Write-Host "=== BATCH RUNNER PID ==="
$pidFile = "C:\Users\Administrator\_batch_runner_pid.txt"
if (Test-Path $pidFile) { $pid = Get-Content $pidFile; Write-Host "Saved PID: $pid"; Get-Process -Id $pid -ErrorAction SilentlyContinue } else { Write-Host "No PID file" }
Write-Host "=== REMOTE LOGS DIR ==="
Get-ChildItem "I:\Github\Latent_Style\SchrodingerBridge\exp\630_remote_logs" -ErrorAction SilentlyContinue
Write-Host "=== DONE ==="
