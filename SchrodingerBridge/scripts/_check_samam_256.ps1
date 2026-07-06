# Check SaMam 256 eval results
Write-Host "=== SaMam 256 eval JSON ==="
Get-ChildItem "I:\exp_256_photo2art" -Filter "eval_samam*.json" -ErrorAction SilentlyContinue | ForEach-Object {
    Write-Host "--- $($_.Name) ---"
    Get-Content $_.FullName -Raw
}

Write-Host "`n=== exp/baseline_v2 SaMam dirs ==="
$samamDirs = Get-ChildItem "I:\Github\Latent_Style\SchrodingerBridge\exp\baseline_v2" -Directory -Recurse -Filter "*samam*" -ErrorAction SilentlyContinue -Depth 3
foreach ($d in $samamDirs) { Write-Host "  $($d.FullName)" }

Write-Host "`n=== final_works/SaMam ==="
$samamFinal = "I:\Github\Latent_Style\final_works\SaMam"
if (Test-Path $samamFinal) {
    Get-ChildItem $samamFinal -Recurse -Depth 2 | ForEach-Object { Write-Host "  $($_.FullName)" }
}

Write-Host "`n=== Search SaMam summary.json ==="
$samamSummaries = Get-ChildItem "I:\Github\Latent_Style" -Recurse -Filter "summary.json" -ErrorAction SilentlyContinue -Depth 5 | Where-Object { $_.DirectoryName -match "SaMam|samam" }
foreach ($s in $samamSummaries) { Write-Host "  $($s.FullName)" }
