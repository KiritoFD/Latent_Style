$gpu = & "C:\Program Files\NVIDIA Corporation\NVSMI\nvidia-smi.exe" --query-gpu=utilization.gpu,memory.used,memory.total,power.draw,temperature.gpu --format=csv,noheader 2>$null
if (-not $gpu) {
    $gpu = & nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total,power.draw,temperature.gpu --format=csv,noheader 2>$null
}
Write-Host "GPU: $gpu"
$p = Get-Process python -ErrorAction SilentlyContinue
if ($p) {
    foreach ($proc in $p) {
        Write-Host "PID=$($proc.Id) CPU=$($proc.CPU)s WS=$([math]::Round($proc.WS/1MB,0))MB"
    }
} else {
    Write-Host "No python process"
}
