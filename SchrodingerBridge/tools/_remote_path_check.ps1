$ErrorActionPreference = "SilentlyContinue"
Write-Host "===CONFIGS DIR==="
Get-ChildItem "I:\Github\Latent_Style\SchrodingerBridge\configs" | ForEach-Object { Write-Host $_.Name }
Write-Host "===STATE DIR==="
Get-ChildItem "I:\Github\Latent_Style\SchrodingerBridge\docs\630\state" | ForEach-Object { Write-Host $_.Name $_.Length }
Write-Host "===I: ROOT==="
Get-ChildItem "I:\" | ForEach-Object { Write-Host $_.Name }
Write-Host "===WIKIART SEARCH==="
Get-ChildItem "I:\" -Directory -Recurse -Depth 2 -Filter "wikiart*" | ForEach-Object { Write-Host $_.FullName }
Write-Host "===EXP DIRS==="
Get-ChildItem "I:\Github\Latent_Style\SchrodingerBridge\exp" | ForEach-Object { Write-Host $_.Name }
Write-Host "===DONE==="
