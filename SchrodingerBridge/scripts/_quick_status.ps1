# Quick status check
$pyProcs = Get-Process -Name python -ErrorAction SilentlyContinue
Write-Output "Python processes: $($pyProcs.Count)"
foreach ($p in $pyProcs) {
    Write-Output "  PID=$($p.Id) Mem=$([math]::Round($p.WorkingSet64/1MB))MB"
}

# R5 count
$r5 = (Get-ChildItem "I:\Github\Latent_Style\SchrodingerBridge\exp\baseline_stylealigned\random5\images" -File -ErrorAction SilentlyContinue | Measure-Object).Count
Write-Output "R5 images: $r5"

# GPU
nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu --format=csv,noheader

# Check launcher log
if (Test-Path "C:\Users\Administrator\logs\zstar_launcher.log") {
    Write-Output "Launcher log:"
    Get-Content "C:\Users\Administrator\logs\zstar_launcher.log" -Tail 5
}
