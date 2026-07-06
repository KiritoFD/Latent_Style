# Check pipeline progress
$ErrorActionPreference = "Continue"

Write-Host "=== Running processes ==="
Get-CimInstance Win32_Process -Filter "Name='python.exe' OR Name='powershell.exe'" |
    Select-Object ProcessId, Name, @{N='Start';E={$_.CreationDate}}, @{N='CPU';E={$_.UserModeTime/1e7}} |
    Format-Table -Auto

Write-Host ""
Write-Host "=== Log files ==="
Get-ChildItem "I:\Github\Latent_Style\SchrodingerBridge\logs" -Filter "*eval*" -ErrorAction SilentlyContinue |
    Sort-Object LastWriteTime -Descending |
    Select-Object Name, Length, LastWriteTime |
    Format-Table -Auto

Get-ChildItem "I:\Github\Latent_Style\SchrodingerBridge\logs" -Filter "master*" -ErrorAction SilentlyContinue |
    Sort-Object LastWriteTime -Descending |
    Select-Object Name, Length, LastWriteTime |
    Format-Table -Auto

Write-Host ""
Write-Host "=== master_pipeline.log tail ==="
$mlog = "I:\Github\Latent_Style\SchrodingerBridge\logs\master_pipeline.log"
if (Test-Path $mlog) { Get-Content $mlog -Tail 30 }

Write-Host ""
Write-Host "=== eval_all_unified.log tail ==="
$elog = "I:\Github\Latent_Style\SchrodingerBridge\logs\eval_all_unified.log"
if (Test-Path $elog) { Get-Content $elog -Tail 30 }

Write-Host ""
Write-Host "=== Existing eval JSON results ==="
Get-ChildItem "I:\Github\Latent_Style\SchrodingerBridge\exp" -Filter "_eval_*.json" -ErrorAction SilentlyContinue |
    Select-Object Name, Length, LastWriteTime |
    Format-Table -Auto

Write-Host ""
Write-Host "=== VRAM ==="
nvidia-smi --query-gpu=memory.used,memory.free,memory.total,utilization.gpu --format=csv
