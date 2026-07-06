# Find WEAVE image directories for D5 and W20 MUSIQ eval
$ErrorActionPreference = "Continue"

Write-Host "=== Search for WEAVE/FCSB images ==="
$locations = @(
    "I:\Github\Latent_Style\SchrodingerBridge\exp\FCSB",
    "I:\Github\Latent_Style\SchrodingerBridge\exp\630_random20_heun_5ep",
    "I:\Github\Latent_Style\SchrodingerBridge\exp\baseline_v2\images"
)

foreach ($loc in $locations) {
    if (Test-Path $loc) {
        Write-Host "--- $loc ---"
        Get-ChildItem $loc -Directory -ErrorAction SilentlyContinue | Select-Object -First 10 Name
    }
}

Write-Host ""
Write-Host "=== Look for weave/weave/ours images ==="
Get-ChildItem "I:\Github\Latent_Style\SchrodingerBridge\exp" -Recurse -Directory -ErrorAction SilentlyContinue |
    Where-Object { $_.Name -like "*weave*" -or $_.Name -like "*ours*" -or $_.Name -like "*fcsb*" } |
    Select-Object -First 10 FullName

Write-Host ""
Write-Host "=== baseline_v2/images all subdirs ==="
$base = "I:\Github\Latent_Style\SchrodingerBridge\exp\baseline_v2\images"
if (Test-Path $base) {
    Get-ChildItem $base -Directory | ForEach-Object {
        $cnt = (Get-ChildItem $_.FullName -File -ErrorAction SilentlyContinue | Measure-Object).Count
        Write-Host ("{0}: {1}" -f $_.Name, $cnt)
    }
}
