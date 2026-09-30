$ErrorActionPreference = 'SilentlyContinue'

Write-Host "===== Batch runner scripts ====="
$bats = @(
    "I:\Github\Latent_Style\SchrodingerBridge\_628_p7_runner.bat",
    "I:\Github\Latent_Style\SchrodingerBridge\_628_p8d_launcher.bat"
)
foreach ($b in $bats) {
    Write-Host "--- $b ---"
    if (Test-Path $b) {
        Get-Content $b
    } else {
        Write-Host "  (not found)"
    }
    Write-Host ""
}

Write-Host "===== 628_run_destructive_batch.py (first 80 lines) ====="
$bp = "I:\Github\Latent_Style\SchrodingerBridge\628_run_destructive_batch.py"
if (Test-Path $bp) {
    Get-Content $bp -TotalCount 80
}

Write-Host ""
Write-Host "===== D17/D18 config files ====="
$cfgs = @(
    "I:\Github\Latent_Style\SchrodingerBridge\configs\ablations\628_destructive\D17_skip_residual_0.json",
    "I:\Github\Latent_Style\SchrodingerBridge\configs\ablations\628_destructive\D18_kinetic_off.json"
)
foreach ($c in $cfgs) {
    Write-Host "--- $c ---"
    if (Test-Path $c) { Get-Content $c } else { Write-Host "  (not found)" }
    Write-Host ""
}

Write-Host "===== All configs in 628_destructive ====="
Get-ChildItem "I:\Github\Latent_Style\SchrodingerBridge\configs\ablations\628_destructive\" -Filter *.json |
    Select-Object Name, LastWriteTime | Format-Table -AutoSize

Write-Host "===== Any progress / log files for 628 ====="
Get-ChildItem "I:\Github\Latent_Style\SchrodingerBridge\" -Filter "*628*" -ErrorAction SilentlyContinue |
    Select-Object Name, Length, LastWriteTime | Format-Table -AutoSize
Get-ChildItem "I:\Github\Latent_Style\SchrodingerBridge\exp\" -Filter "*628*" -Recurse -ErrorAction SilentlyContinue -Depth 2 |
    Select-Object FullName, LastWriteTime | Format-Table -AutoSize

Write-Host "===== Output logs from the two batch runs ====="
# Look for stdout logs near the launcher bats or in a likely logs dir
$logDirs = @(
    "I:\Github\Latent_Style\SchrodingerBridge\logs",
    "I:\Github\Latent_Style\SchrodingerBridge\exp\628_destructive"
)
foreach ($d in $logDirs) {
    if (Test-Path $d) {
        Write-Host "--- $d ---"
        Get-ChildItem $d -Recurse -ErrorAction SilentlyContinue | Sort-Object LastWriteTime -Descending | Select-Object -First 20 FullName, Length, LastWriteTime | Format-Table -AutoSize
    }
}
