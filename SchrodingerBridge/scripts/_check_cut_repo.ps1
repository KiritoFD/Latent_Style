# Inspect CUT repo structure
Write-Host "=== CUT repo structure ==="
$cutRepo = "I:\Github\Latent_Style\Related_Works\repos\external\CUT"
if (Test-Path $cutRepo) {
    Get-ChildItem $cutRepo | ForEach-Object { Write-Host "  $($_.Name)" }
    Write-Host "`n--- models dir ---"
    if (Test-Path "$cutRepo\models") {
        Get-ChildItem "$cutRepo\models" -Filter "*.py" | ForEach-Object { Write-Host "  $($_.Name)" }
    }
    Write-Host "`n--- options dir ---"
    if (Test-Path "$cutRepo\options") {
        Get-ChildItem "$cutRepo\options" -Filter "*.py" | ForEach-Object { Write-Host "  $($_.Name)" }
    }
}

Write-Host "`n=== Look for existing CUT gen scripts in our scripts dir ==="
$cutGenScripts = Get-ChildItem "I:\Github\Latent_Style\SchrodingerBridge\scripts" -Filter "*.py" -ErrorAction SilentlyContinue | Where-Object { (Get-Content $_.FullName -TotalCount 5 -ErrorAction SilentlyContinue) -match "CUT|cut" }
foreach ($s in $cutGenScripts) { Write-Host "  $($s.Name) ($($s.Length)B)" }

Write-Host "`n=== Look for CUT-related scripts in baseline_pipeline ==="
$bpDir = "I:\Github\Latent_Style\Related_Works\baseline_pipeline"
if (Test-Path $bpDir) {
    Get-ChildItem $bpDir -Filter "*.py" -Recurse -ErrorAction SilentlyContinue | Where-Object { $_.Name -match "cut|CUT" } | ForEach-Object { Write-Host "  $($_.FullName) ($($_.Length)B)" }
}

Write-Host "`n=== Check exp_baselines/cut ==="
$expCut = "I:\Github\Latent_Style\exp_baselines\cut"
if (Test-Path $expCut) {
    Get-ChildItem $expCut -Recurse -ErrorAction SilentlyContinue | Select-Object -First 20 | ForEach-Object { Write-Host "  $($_.FullName)" }
}
