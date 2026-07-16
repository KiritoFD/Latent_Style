# List eval_bundle files precisely
$dir = 'I:\Github\Latent_Style\exp_baselines\samst_distinct5_512_wsl_stepalign40_remote_20260605_r1\eval_bundle'
if (Test-Path $dir) {
    Write-Host "=== eval_bundle contents ==="
    Get-ChildItem $dir -File | Select-Object Name, Length | Format-Table -AutoSize
}

# Read bundle_summary.json
$bundle = Join-Path $dir 'bundle_summary.json'
if (Test-Path $bundle) {
    Write-Host "`n=== bundle_summary.json (first 50 lines) ==="
    Get-Content $bundle | Select-Object -First 50
}

# Read eval_step CSVs
$csvs = Get-ChildItem $dir -Filter "eval_step_*.csv" -ErrorAction SilentlyContinue
foreach ($csv in $csvs) {
    Write-Host "`n=== $($csv.Name) ==="
    Get-Content $csv.FullName | Select-Object -First 10
}

# Read eval_step JSONs (first 30 lines each)
$jsons = Get-ChildItem $dir -Filter "eval_step_*.json" -ErrorAction SilentlyContinue
foreach ($j in $jsons) {
    Write-Host "`n=== $($j.Name) (first 30 lines) ==="
    Get-Content $j.FullName | Select-Object -First 30
}
