# Find 256 CUT checkpoints (cezanne/Hayao/monet/photo/vangogh)
Write-Host "=== Search for cut_to_cezanne / cut_to_Hayao etc (256) ==="
$found = Get-ChildItem "I:\Github\Latent_Style" -Recurse -Directory -Filter "cut_to_*" -ErrorAction SilentlyContinue -Depth 6
foreach ($f in $found) {
    $ckpts = Get-ChildItem $f.FullName -Filter "*.pth" -ErrorAction SilentlyContinue
    if ($ckpts.Count -gt 0) {
        Write-Host "  $($f.FullName)"
        foreach ($c in $ckpts) { Write-Host "    $($c.Name) ($([math]::Round($c.Length/1MB,1))MB)" }
    }
}

Write-Host "`n=== exp_baselines/_auxiliary_runs/cut_5x5/cut_repo/infer_5x5 ==="
$auxCut = "I:\Github\Latent_Style\exp_baselines\_auxiliary_runs\cut_5x5\cut_repo\infer_5x5"
if (Test-Path $auxCut) {
    Get-ChildItem $auxCut -Recurse -ErrorAction SilentlyContinue | Select-Object -First 30 | ForEach-Object { Write-Host "  $($_.FullName)" }
}

Write-Host "`n=== final_works/CUT/logs/test_cut_to_cezanne.log (last 10 lines) ==="
$logFile = "I:\Github\Latent_Style\final_works\CUT\logs\test_cut_to_cezanne.log"
if (Test-Path $logFile) {
    Get-Content $logFile -Tail 15
}
