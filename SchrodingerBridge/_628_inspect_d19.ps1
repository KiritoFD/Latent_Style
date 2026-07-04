# Inspect D19 log to find why the second batch crashed
$logPath = 'I:/Github/Latent_Style/SchrodingerBridge/exp/628_ablation/destructive_logs/D19_attn_gated_raw.log'
if (Test-Path $logPath) {
    Write-Host "=== D19 log tail (last 40 lines) ==="
    Get-Content $logPath -Tail 40
} else {
    Write-Host "D19 log not found: $logPath"
}

Write-Host "`n=== Current GPU state ==="
$pyProcs = Get-Process python -ErrorAction SilentlyContinue
if ($pyProcs) {
    foreach ($p in $pyProcs) {
        Write-Host "PID=$($p.Id) StartTime=$($p.StartTime) CPU=$($p.CPU)"
    }
} else {
    Write-Host "No python process running (GPU is free)"
}

Write-Host "`n=== nvidia-smi ==="
$nv = Get-Command nvidia-smi -ErrorAction SilentlyContinue
if ($nv) {
    & nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu --format=csv
}
