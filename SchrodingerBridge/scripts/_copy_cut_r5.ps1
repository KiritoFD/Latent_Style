$src = "g:\GitHub\Latent_Style\SchrodingerBridge\results\D5-512\cut"
$dst = "g:\GitHub\Latent_Style\SchrodingerBridge\results\R5-WikiArt\cut"

if (-not (Test-Path $dst)) {
    New-Item -ItemType Directory -Path $dst -Force | Out-Null
}

$existing = (Get-ChildItem $dst -File -ErrorAction SilentlyContinue).Count
if ($existing -gt 0) {
    Write-Host "R5-WikiArt/cut already has $existing files, skipping copy"
    exit 0
}

Write-Host "Copying D5-512/cut -> R5-WikiArt/cut (same images per _eval_cut_w20.json)"
Copy-Item -Path "$src\*.png" -Destination $dst -Force
$cnt = (Get-ChildItem $dst -Filter *.png -ErrorAction SilentlyContinue).Count
Write-Host "Copied: $cnt png files"
