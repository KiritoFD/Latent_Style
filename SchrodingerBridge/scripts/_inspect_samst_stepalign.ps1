# Inspect samst stepalign40 directory
$dir = 'I:\Github\Latent_Style\exp_baselines\samst_distinct5_512_wsl_stepalign40_remote_20260605_r1'
if (Test-Path $dir) {
    Write-Host "=== Directory tree (depth 2) ==="
    Get-ChildItem $dir -Depth 2 | Select-Object FullName, Length | Format-Table -AutoSize
    
    # Look for summary.json
    $summary = Get-ChildItem $dir -Recurse -Filter "summary.json" -ErrorAction SilentlyContinue | Select-Object -First 1
    if ($summary) {
        Write-Host "=== summary.json ==="
        Get-Content $summary.FullName
    }
    
    # Look for any CSV
    $csvs = Get-ChildItem $dir -Recurse -Filter "*.csv" -ErrorAction SilentlyContinue
    if ($csvs) {
        Write-Host "`n=== CSV files ==="
        $csvs | Select-Object FullName, Length | Format-Table -AutoSize
    }
    
    # Look for any JSON
    $jsons = Get-ChildItem $dir -Recurse -Filter "*.json" -ErrorAction SilentlyContinue
    if ($jsons) {
        Write-Host "`n=== JSON files ==="
        $jsons | Select-Object FullName, Length | Format-Table -AutoSize
    }
}
