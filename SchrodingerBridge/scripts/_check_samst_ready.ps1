$scripts = @(
    'C:\Users\Administrator\_samst_curve_repro.py',
    'C:\Users\Administrator\samst_repo\networks\transfer_net.py'
)
foreach ($s in $scripts) {
    if (Test-Path $s) { Write-Host "OK: $s" } else { Write-Host "MISSING: $s" }
}
Write-Host "--- samst ckpts ---"
$ckptRoot = 'C:\Users\Administrator\samst_ckpts'
if (Test-Path $ckptRoot) {
    Get-ChildItem $ckptRoot -Recurse -Filter "*.model" | ForEach-Object { Write-Host $_.FullName }
} else {
    Write-Host "ckpt root missing"
}
Write-Host "--- samst script on remote ---"
if (Test-Path 'C:\Users\Administrator\_samst_curve_repro.py') {
    Write-Host "OK samst script"
} else {
    Write-Host "MISSING samst script"
}
