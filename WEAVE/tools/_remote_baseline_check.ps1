$ErrorActionPreference = "SilentlyContinue"
Write-Host "===BASELINE V2 DIR==="
Get-ChildItem "I:\Github\Latent_Style\SchrodingerBridge\exp\baseline_v2" -Recurse -Depth 1 | ForEach-Object { Write-Host $_.FullName.Replace("I:\Github\Latent_Style\SchrodingerBridge\exp\baseline_v2\","") }
Write-Host "===BASELINE CONFIG==="
$cfgFiles = Get-ChildItem "I:\Github\Latent_Style\SchrodingerBridge\exp\baseline_v2" -Filter "*.json" -Recurse
foreach ($f in $cfgFiles) { Write-Host "--- $($f.Name) ---"; Get-Content $f.FullName -Head 50 }
Write-Host "===TRAINING DATA==="
Get-ChildItem "I:\wikiart_distinct5_samam_512_latents_ema" | ForEach-Object { Write-Host $_.Name }
Write-Host "===TRAIN SUBDIRS==="
if (Test-Path "I:\wikiart_distinct5_samam_512_latents_ema\train") { Get-ChildItem "I:\wikiart_distinct5_samam_512_latents_ema\train" | ForEach-Object { Write-Host $_.Name } }
Write-Host "===LATENT CACHE==="
if (Test-Path "I:\wikiart_distinct5_samam_512_latents_ema\train\.latent_cache") { Get-ChildItem "I:\wikiart_distinct5_samam_512_latents_ema\train\.latent_cache" -Recurse | ForEach-Object { Write-Host $_.FullName.Replace("I:\wikiart_distinct5_samam_512_latents_ema\train\.latent_cache\","") } }
Write-Host "===TEST DIR==="
Get-ChildItem "I:\wikiart_distinct5_samam_512_classview\test" | ForEach-Object { Write-Host $_.Name }
Write-Host "===CLEAN_BASE CONFIG (for reference)==="
$cbCfg = "I:\Github\Latent_Style\SchrodingerBridge\configs\clean_base.json"
if (Test-Path $cbCfg) { Get-Content $cbCfg -Head 30 }
Write-Host "===DONE==="
