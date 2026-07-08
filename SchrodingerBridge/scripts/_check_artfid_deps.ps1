$ErrorActionPreference = 'Continue'

# Check art_inception checkpoint
Write-Host "=== Checking art_inception checkpoint ==="
$cache_dir = "G:\GitHub\Latent_Style\eval_cache\artfid"
if (Test-Path "$cache_dir\art_inception.pth") {
    $size = (Get-Item "$cache_dir\art_inception.pth").Length
    Write-Host "Found: $cache_dir\art_inception.pth ($([math]::Round($size/1MB,2)) MB)"
} else {
    Write-Host "Not found at $cache_dir"
}

Write-Host ""
Write-Host "=== Check metrics.csv for each P256 method ==="
foreach ($m in @("identity_256", "adain_256", "wct_256", "sdturbo_256", "samst_256", "samam_256", "styleid_256")) {
    $p = "I:\exp_256_photo2art\$m\metrics.csv"
    $exists = ssh -p 2222 -o LogLevel=ERROR administrator@100.115.18.62 "if exist $p (echo EXISTS) else (echo MISSING)"
    Write-Host "${m}: $exists"
}

Write-Host ""
Write-Host "=== Check metrics.csv for D5-512 methods ==="
foreach ($m in @("identity", "adain", "wct_v32k", "sdturbo", "cut", "samst", "samam", "styleid")) {
    $p = "I:\Github\Latent_Style\SchrodingerBridge\exp\baseline_v2\images\$m\metrics.csv"
    $exists = ssh -p 2222 -o LogLevel=ERROR administrator@100.115.18.62 "if exist `"$p`" (echo EXISTS) else (echo MISSING)"
    Write-Host "${m}: $exists"
}

Write-Host ""
Write-Host "=== Check metrics.csv for R5-WikiArt methods ==="
foreach ($m in @("identity", "adain", "wct", "sdturbo", "samst", "samam", "styleid")) {
    $p = "I:\Github\Latent_Style\SchrodingerBridge\exp\baseline_wikiarts20\$m\metrics.csv"
    $exists = ssh -p 2222 -o LogLevel=ERROR administrator@100.115.18.62 "if exist `"$p`" (echo EXISTS) else (echo MISSING)"
    Write-Host "${m}: $exists"
}

Write-Host ""
Write-Host "=== Check metrics.csv for WEAVE paths ==="
$weave_paths = @(
    "I:\Github\Latent_Style\SchrodingerBridge\exp\clean_base_v2\full_eval\epoch_0010\metrics.csv",
    "I:\Github\Latent_Style\SchrodingerBridge\exp\latent256_photo2art\latent256_b16_e10\full_eval\epoch_0010\metrics.csv",
    "I:\Github\Latent_Style\SchrodingerBridge\exp\wikiarts20_eval\metrics.csv"
)
foreach ($p in $weave_paths) {
    $exists = ssh -p 2222 -o LogLevel=ERROR administrator@100.115.18.62 "if exist `"$p`" (echo EXISTS) else (echo MISSING)"
    Write-Host "${p}: $exists"
}

Write-Host ""
Write-Host "=== Check metrics.csv for Seedream D5 ==="
$ssh_out = ssh -p 2222 -o LogLevel=ERROR administrator@100.115.18.62 "dir /B /S I:\Github\Latent_Style\exp_baselines\seedream45_api\distinct5_512_seedream45_windhub_20260607_repaired750\*.csv 2>nul"
Write-Host $ssh_out
