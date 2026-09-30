$ErrorActionPreference = "SilentlyContinue"
Write-Output "=== PYTHON PROCESSES ==="
$procs = Get-Process python
if ($procs) {
    $procs | Select-Object Id,@{N="WS_MB";E={[math]::Round($_.WorkingSet/1MB,1)}} | Format-Table -AutoSize
} else {
    Write-Output "No python processes running"
}
Write-Output "=== T11_REPRO SUMMARY ==="
$sr = Get-ChildItem "I:\Github\Latent_Style\SchrodingerBridge\exp\t11_repro_15ep\full_eval" -Recurse -Filter "summary.json"
if ($sr) { $sr | ForEach-Object { Write-Output $_.FullName } } else { Write-Output "No summary.json found" }
Write-Output "=== T11E2 SUMMARY ==="
$se = Get-ChildItem "I:\Github\Latent_Style\SchrodingerBridge\exp\t11e2_extrap05_15ep\full_eval" -Recurse -Filter "summary.json"
if ($se) { $se | ForEach-Object { Write-Output $_.FullName } } else { Write-Output "No summary.json found" }
Write-Output "=== T11_REPRO CHECKPOINTS ==="
Get-ChildItem "I:\Github\Latent_Style\SchrodingerBridge\exp\t11_repro_15ep" -Filter "*.pt" | Select-Object Name,@{N="MB";E={[math]::Round($_.Length/1MB,1)}} | Format-Table -AutoSize
Write-Output "=== T11E2 CHECKPOINTS ==="
Get-ChildItem "I:\Github\Latent_Style\SchrodingerBridge\exp\t11e2_extrap05_15ep" -Filter "*.pt" | Select-Object Name,@{N="MB";E={[math]::Round($_.Length/1MB,1)}} | Format-Table -AutoSize
Write-Output "=== DONE ==="
