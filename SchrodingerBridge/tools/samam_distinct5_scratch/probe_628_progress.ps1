$ErrorActionPreference = 'SilentlyContinue'
$logDir = "I:\Github\Latent_Style\SchrodingerBridge\exp\628_ablation\destructive_logs"

Write-Host "===== Batch log (tail 80) ====="
$bl = Join-Path $logDir "batch_log.txt"
if (Test-Path $bl) {
    Get-Content $bl -Tail 80
} else { Write-Host "(no batch_log.txt)" }

Write-Host ""
Write-Host "===== p7_runner.log (tail 30) ====="
$p7 = Join-Path $logDir "p7_runner.log"
if (Test-Path $p7) { Get-Content $p7 -Tail 30 } else { Write-Host "(no p7_runner.log)" }

Write-Host ""
Write-Host "===== p8d_launcher.log (tail 30) ====="
$p8d = Join-Path $logDir "p8d_launcher.log"
if (Test-Path $p8d) { Get-Content $p8d -Tail 30 } else { Write-Host "(no p8d_launcher.log)" }

Write-Host ""
Write-Host "===== Completed experiments (epoch_0010.pt markers) ====="
$done = Get-ChildItem "I:\Github\Latent_Style\SchrodingerBridge\exp\628_ablation\destructive\" -Recurse -Filter "epoch_0010.pt" -ErrorAction SilentlyContinue
Write-Host "Completed count: $($done.Count)"
$done | Select-Object FullName, LastWriteTime | Sort-Object LastWriteTime | Format-Table -AutoSize

Write-Host ""
Write-Host "===== Currently running: D17 and D18 log tails ====="
foreach ($n in @("D17_skip_residual_0","D18_kinetic_off")) {
    $lp = Join-Path $logDir "$n.log"
    if (Test-Path $lp) {
        Write-Host "--- $n.log (tail 15) ---"
        Get-Content $lp -Tail 15
    } else { Write-Host "(no $n.log)" }
}

Write-Host ""
Write-Host "===== Total configs to run ====="
$total = (Get-ChildItem "I:\Github\Latent_Style\SchrodingerBridge\configs\ablations\628_destructive\" -Filter *.json).Count
Write-Host "Total configs: $total"
