# Check SaMam Random5 status
$ErrorActionPreference = "Continue"
$REPO = "I:\Github\Latent_Style\SchrodingerBridge"

Write-Host "=== SaMam images count ==="
$dir = "$REPO\exp\baseline_wikiarts20\samam\images"
if (Test-Path $dir) {
    $cnt = (Get-ChildItem $dir -File).Count
    Write-Host "samam: $cnt / 750"
}

Write-Host ""
Write-Host "=== SaMam Random5 fp16 log ==="
$log = "$REPO\logs\samam_random5_fp16.log"
if (Test-Path $log) {
    Get-Content $log -Tail 30
}

Write-Host ""
Write-Host "=== Schtasks status ==="
schtasks /Query /TN "samam_random5" /FO LIST 2>$null | Select-Object -First 8

Write-Host ""
Write-Host "=== Check _DONE marker ==="
$done = "$REPO\exp\baseline_wikiarts20\samam\_DONE"
if (Test-Path $done) {
    Write-Host "DONE marker found:"
    Get-Content $done
} else {
    Write-Host "No DONE marker"
}
