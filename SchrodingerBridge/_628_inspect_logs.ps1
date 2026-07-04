# Inspect training logs to verify ablation actually applied
$ErrorActionPreference = 'Continue'
$root = 'I:/Github/Latent_Style/SchrodingerBridge'
$logDir = "$root\exp\628_ablation\destructive_logs"

Write-Host "=== D10_style_gate_film_only log (first 80 lines) ==="
$d10Log = Join-Path $logDir 'D10_style_gate_film_only.log'
if (Test-Path $d10Log) {
    Get-Content $d10Log -Head 80 | ForEach-Object { Write-Host "  $_" }
} else {
    Write-Host "  MISSING"
}

Write-Host "`n=== D10 log: search for style_gate_mode / film_only ==="
if (Test-Path $d10Log) {
    Select-String -Path $d10Log -Pattern 'style_gate_mode|film_only|gate|ablation' -SimpleMatch | Select-Object -First 15 | ForEach-Object { Write-Host "  L$($_.LineNumber): $($_.Line)" }
}

Write-Host "`n=== D1_spectral_ode_off log (key lines) ==="
$d1Log = Join-Path $logDir 'D1_spectral_ode_off.log'
if (Test-Path $d1Log) {
    Select-String -Path $d1Log -Pattern 'contract_family|spectral_ode|620_spatial|ablation' -SimpleMatch | Select-Object -First 15 | ForEach-Object { Write-Host "  L$($_.LineNumber): $($_.Line)" }
}

Write-Host "`n=== D2_adain_scale_0 log (key lines) ==="
$d2Log = Join-Path $logDir 'D2_adain_scale_0.log'
if (Test-Path $d2Log) {
    Select-String -Path $d2Log -Pattern 'endpoint_adain_scale|adain|ablation' -SimpleMatch | Select-Object -First 15 | ForEach-Object { Write-Host "  L$($_.LineNumber): $($_.Line)" }
}

Write-Host "`n=== Verify config content: D10 ==="
$d10Cfg = "$root\configs\ablations\628_destructive\D10_style_gate_film_only.json"
if (Test-Path $d10Cfg) {
    $j = Get-Content $d10Cfg -Raw | ConvertFrom-Json
    Write-Host "  model.style_gate_mode = $($j.model.style_gate_mode)"
    Write-Host "  ablation.notes = $($j.ablation.notes)"
}
