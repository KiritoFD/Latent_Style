# Sync modified spectral_losses620.py, stop batch, clean stale results
$root = 'I:/Github/Latent_Style/SchrodingerBridge'
$remoteDst = "$root/src/spectral_losses620.py"
$python = 'C:\Progra~1\Python312\python.exe'

# File already copied via scp to /tmp/spectral_losses620.py
Copy-Item /tmp/spectral_losses620.py $remoteDst -Force
Write-Host "Copied: /tmp/spectral_losses620.py -> $remoteDst"

# Verify syntax
$checkScript = "import ast; ast.parse(open(r'$remoteDst', encoding='utf-8').read()); print('Remote syntax OK')"
& $python -c $checkScript
if ($LASTEXITCODE -ne 0) {
    Write-Host "SYNTAX CHECK FAILED - aborting"
    exit 1
}

# 1. Stop batch runner + watchdog
Write-Host "`n=== Stopping batch runner and watchdog ==="
schtasks /End /TN 'sb_628_batch_runner' 2>$null
schtasks /End /TN 'sb_628_watchdog' 2>$null
Start-Sleep -Seconds 3

# Kill lingering python processes
Get-Process python -ErrorAction SilentlyContinue | ForEach-Object {
    Write-Host "Killing python PID=$($_.Id)"
    Stop-Process -Id $_.Id -Force -ErrorAction SilentlyContinue
}
Start-Sleep -Seconds 2

# 2. Clean stale E1-E24 + L1-L6 + L11-L16 results (used zero-placeholder losses)
Write-Host "`n=== Cleaning stale results ==="
$expDir = "$root\exp\628_ablation\destructive"
$stalePatterns = @(
    'E1_*', 'E2_*', 'E3_*', 'E4_*', 'E5_*', 'E6_*', 'E7_*', 'E8_*', 'E9_*',
    'E10_*', 'E11_*', 'E12_*', 'E13_*', 'E14_*', 'E15_*', 'E16_*',
    'E17_*', 'E18_*', 'E19_*', 'E20_*', 'E21_*', 'E22_*', 'E23_*', 'E24_*',
    'L1_*', 'L2_*', 'L3_*', 'L4_*', 'L5_*', 'L6_*',
    'L11_*', 'L12_*', 'L13_*', 'L14_*', 'L15_*', 'L16_*'
)
# Keep: D1-D30 (architecture), L7-L10 (spectral_w_* real), P1-P18 (param sweeps)

$cleanedCount = 0
foreach ($pattern in $stalePatterns) {
    $dirs = Get-ChildItem $expDir -Directory -Filter $pattern -ErrorAction SilentlyContinue
    foreach ($d in $dirs) {
        Write-Host "  Removing: $($d.Name)"
        Remove-Item $d.FullName -Recurse -Force -ErrorAction SilentlyContinue
        $cleanedCount++
    }
}
Write-Host "Cleaned $cleanedCount stale experiment directories"

# 3. Verify remaining
Write-Host "`n=== Remaining experiments ==="
$remaining = Get-ChildItem $expDir -Directory -ErrorAction SilentlyContinue
Write-Host "Remaining: $($remaining.Count) experiments"
$remaining | ForEach-Object { Write-Host "  $($_.Name)" }

# 4. GPU state
Write-Host "`n=== GPU state ==="
& nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu --format=csv
