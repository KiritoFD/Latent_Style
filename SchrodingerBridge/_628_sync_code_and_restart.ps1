# Sync modified spectral_losses620.py to remote, stop current batch, clean stale E1-E24 + L1-L6 results
$root = 'I:/Github/Latent_Style/SchrodingerBridge'
$localSrc = 'g:/GitHub/Latent_Style/SchrodingerBridge/src/spectral_losses620.py'
$remoteDst = 'I:/Github/Latent_Style/SchrodingerBridge/src/spectral_losses620.py'

# 1. Stop current batch runner + watchdog
Write-Host "=== Stopping batch runner and watchdog ==="
schtasks /End /TN 'sb_628_batch_runner' 2>$null
schtasks /End /TN 'sb_628_watchdog' 2>$null
Start-Sleep -Seconds 3

# Kill any lingering python processes (training subprocesses)
Get-Process python -ErrorAction SilentlyContinue | ForEach-Object {
    Write-Host "Killing python PID=$($_.Id)"
    Stop-Process -Id $_.Id -Force -ErrorAction SilentlyContinue
}
Start-Sleep -Seconds 2

# 2. Sync the modified spectral_losses620.py
Write-Host "`n=== Syncing spectral_losses620.py ==="
Copy-Item $localSrc $remoteDst -Force
Write-Host "Copied: $localSrc -> $remoteDst"

# Verify syntax on remote
$python = 'C:\Progra~1\Python312\python.exe'
& $python -c "import ast; ast.parse(open(r'$remoteDst', encoding='utf-8').read()); print('Remote syntax OK')"

# 3. Clean stale E1-E24 + L1-L6 + L13-L16 results (these used zero-placeholder losses)
Write-Host "`n=== Cleaning stale results (E1-E24, L1-L6, L13-L16) ==="
$expDir = "$root\exp\628_ablation\destructive"
$stalePatterns = @('E1_*', 'E2_*', 'E3_*', 'E4_*', 'E5_*', 'E6_*', 'E7_*', 'E8_*', 'E9_*',
                   'E10_*', 'E11_*', 'E12_*', 'E13_*', 'E14_*', 'E15_*', 'E16_*',
                   'E17_*', 'E18_*', 'E19_*', 'E20_*', 'E21_*', 'E22_*', 'E23_*', 'E24_*',
                   'L1_*', 'L2_*', 'L3_*', 'L4_*', 'L5_*', 'L6_*',
                   'L11_*', 'L12_*')  # L11/L12 also used zero-placeholder losses
# Note: L7/L8/L9/L10 use spectral_w_* which ARE real, keep them
# L13/L14/L15/L16 also use zero-placeholder (w_flow, coupling_structure_edge_weight, etc.)
$stalePatterns += @('L13_*', 'L14_*', 'L15_*', 'L16_*')

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

# 4. Verify remaining experiments
Write-Host "`n=== Remaining experiments ==="
$remaining = Get-ChildItem $expDir -Directory -ErrorAction SilentlyContinue
Write-Host "Remaining: $($remaining.Count) experiments"
$remaining | Select-Object -First 10 | ForEach-Object { Write-Host "  $($_.Name)" }
Write-Host "  ..."

# 5. Check GPU is free now
Write-Host "`n=== GPU state ==="
& nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu --format=csv
