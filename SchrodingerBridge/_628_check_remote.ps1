$root = 'I:\Github\Latent_Style\SchrodingerBridge'
Write-Host "=== Checking $root ==="
if (-not (Test-Path $root)) {
    Write-Host "ERROR: $root does not exist"
    exit 1
}

$files = @(
    'src\spectral_losses620.py',
    '_628_gen_extreme_loss_configs.py',
    '_628_start_extreme_batch.ps1',
    '_628_watchdog.ps1',
    '628_run_destructive_batch.py'
)
foreach ($f in $files) {
    $p = Join-Path $root $f
    if (Test-Path $p) {
        $info = Get-Item $p
        Write-Host "OK  $($info.Length) bytes  $f"
    } else {
        Write-Host "MISSING  $f"
    }
}

# Count X configs
$cfgDir = Join-Path $root 'configs\ablations\628_destructive'
$xConfigs = Get-ChildItem $cfgDir -Filter 'X*.json' -ErrorAction SilentlyContinue
Write-Host ""
Write-Host "X configs in $cfgDir : $($xConfigs.Count)"
if ($xConfigs.Count -gt 0) {
    $xConfigs | Select-Object -First 5 | ForEach-Object { Write-Host "  $($_.Name)" }
    Write-Host "  ..."
    $xConfigs | Select-Object -Last 3 | ForEach-Object { Write-Host "  $($_.Name)" }
}

# Verify spectral_losses620.py contains the new code
$lossFile = Join-Path $root 'src\spectral_losses620.py'
if (Test-Path $lossFile) {
    $content = Get-Content $lossFile -Raw
    if ($content -match 'w_velocity_magnitude.*REAL') {
        Write-Host ""
        Write-Host "VERIFY: spectral_losses620.py has REAL auxiliary losses code"
    } elseif ($content -match 'w_velocity_magnitude') {
        Write-Host ""
        Write-Host "VERIFY: spectral_losses620.py has w_velocity_magnitude (need to check if REAL)"
    } else {
        Write-Host ""
        Write-Host "VERIFY FAILED: spectral_losses620.py does NOT have new loss code"
    }
    # Show debug print line
    $debugLine = $content -split "`n" | Where-Object { $_ -match '628-DEBUG' } | Select-Object -First 1
    if ($debugLine) {
        Write-Host "DEBUG LINE: $($debugLine.Trim())"
    }
}
