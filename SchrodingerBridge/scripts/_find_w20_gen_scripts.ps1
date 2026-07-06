# Find all W20 generation scripts and check SDTurbo/StyleID capabilities
$ErrorActionPreference = "Continue"

$REPO = "I:\Github\Latent_Style\SchrodingerBridge"

Write-Host "=== All gen scripts ==="
Get-ChildItem "$REPO\scripts" -Filter "_gen_*" -ErrorAction SilentlyContinue |
    Select-Object Name, Length, LastWriteTime |
    Format-Table -Auto

Write-Host ""
Write-Host "=== post_pipeline.ps1 content (first 80 lines) ==="
$pp = "$REPO\scripts\_post_pipeline.ps1"
if (Test-Path $pp) {
    Get-Content $pp -TotalCount 80
}

Write-Host ""
Write-Host "=== Check for sdturbo/styleid gen scripts ==="
Get-ChildItem "$REPO\scripts" -Filter "*sdturbo*" -ErrorAction SilentlyContinue |
    Select-Object Name
Get-ChildItem "$REPO\scripts" -Filter "*styleid*" -ErrorAction SilentlyContinue |
    Select-Object Name

Write-Host ""
Write-Host "=== Check Related_Works repos ==="
$rw = "I:\Github\Latent_Style\Related_Works\repos"
if (Test-Path $rw) {
    Get-ChildItem $rw -Directory | Select-Object Name
}

Write-Host ""
Write-Host "=== SaMST checkpoint styles ==="
$samstCkpt = Get-ChildItem "I:\Github\Latent_Style\Related_Works\repos\SaMST-main" -Filter "*.model" -Recurse -ErrorAction SilentlyContinue
if ($samstCkpt) {
    $samstCkpt | Select-Object FullName, Length
}

Write-Host ""
Write-Host "=== SaMam checkpoint ==="
$samamCkpt = Get-ChildItem "I:\Github\Latent_Style\Related_Works\repos\SaMam" -Filter "*.ckpt" -Recurse -ErrorAction SilentlyContinue
if ($samamCkpt) {
    $samamCkpt | Select-Object FullName, Length
}

Write-Host ""
Write-Host "=== SD-Turbo/StyleID gen scripts in baseline_v2 ==="
Get-ChildItem "$REPO\exp\baseline_v2" -Filter "*.py" -ErrorAction SilentlyContinue |
    Select-Object Name
Get-ChildItem "$REPO\scripts" -Filter "*baseline*" -ErrorAction SilentlyContinue |
    Select-Object Name
