$ErrorActionPreference = 'SilentlyContinue'
$repo = "I:\Github\Latent_Style\SchrodingerBridge"

Write-Output "=== 1. Stop all python ==="
Get-Process python -ErrorAction SilentlyContinue | Stop-Process -Force
Start-Sleep -Seconds 2

Write-Output "=== 2. Remove PID file ==="
Remove-Item "$repo\remote_ablation_runner.pid" -Force -ErrorAction SilentlyContinue

Write-Output "=== 3. Purge ALL polluted ablation results (projection_dirs cache bug) ==="
$ablDirs = @(
    "abl_baseline","abl_k1_global","abl_blend0","abl_blend1","abl_k64",
    "abl_soft_mask","abl_ll_w0","abl_ll_w1","abl_route_p05","abl_route_p10",
    "abl_sinkhorn","abl_spectral",
    "abl_no_swd_loss","abl_no_dwt_route","abl_no_wct","abl_no_eota"
)
foreach ($d in $ablDirs) {
    $p = Join-Path $repo "exp\$d"
    if (Test-Path $p) {
        Remove-Item $p -Recurse -Force
        Write-Output "  purged exp\$d"
    }
}

Write-Output "=== 4. Purge stale exp logs ==="
Get-ChildItem $repo -Filter "exp_log_abl_*.txt" -File | ForEach-Object {
    Remove-Item $_.FullName -Force
}
Get-ChildItem $repo -Filter "remote_ablation_log.txt" -File | ForEach-Object {
    Remove-Item $_.FullName -Force
}

Write-Output "=== 5. GPU status ==="
nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader

Write-Output "=== 6. Verify src code is the fixed version ==="
$f = Join-Path $repo "src\spectral_losses620.py"
$lines = Get-Content $f
$hasCacheBug = $false
foreach ($l in $lines) {
    if ($l -match "_projection_dir_cache") { $hasCacheBug = $true; break }
}
if ($hasCacheBug) {
    Write-Output "ERROR: _projection_dir_cache still present — code NOT fixed!"
    exit 1
} else {
    Write-Output "OK: _projection_dir_cache removed (fresh random dirs)"
}

Write-Output "=== 7. Verify run_remote_ablation.py is the clean version ==="
$runner = Join-Path $repo "run_remote_ablation.py"
$content = Get-Content $runner -Raw
if ($content -match "expandable_segments") {
    Write-Output "ERROR: expandable_segments still in runner!"
    exit 1
} else {
    Write-Output "OK: clean runner (no expandable_segments)"
}

Write-Output ""
Write-Output "=== DEPLOY READY ==="
