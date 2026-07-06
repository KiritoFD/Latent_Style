$log = "I:\Github\Latent_Style\SchrodingerBridge\logs\baseline_wikiarts20.log"
Write-Output "=== last 50 lines of $log ==="
if (Test-Path $log) {
    Get-Content $log -Tail 50
}

Write-Output ""
Write-Output "=== adain eval out (last 30) ==="
$out = "$log.adain.eval.out"
if (Test-Path $out) { Get-Content $out -Tail 30 } else { Write-Output "  (not found)" }

Write-Output ""
Write-Output "=== adain eval err (last 30) ==="
$err = "$log.adain.eval.err"
if (Test-Path $err) { Get-Content $err -Tail 30 } else { Write-Output "  (not found)" }

Write-Output ""
Write-Output "=== identity eval out (last 15) ==="
$iout = "$log.identity.eval.out"
if (Test-Path $iout) { Get-Content $iout -Tail 15 } else { Write-Output "  (not found)" }

Write-Output ""
Write-Output "=== schtasks ==="
schtasks /Query /TN "wikiarts20_all" /FO LIST 2>&1 | Select-String -Pattern "Status|Last"
