# Check evaluation progress for all 3 methods
$script = @'
$repo = "I:\Github\Latent_Style\SchrodingerBridge"
$logMain = "$repo\logs\baseline_wikiarts15.log"

Write-Output "=== main log (last 40 lines) ==="
if (Test-Path $logMain) {
    Get-Content $logMain -Tail 40
}

Write-Output ""
Write-Output "=== output dirs (check for summary.json) ==="
$outRoot = "$repo\exp\baseline_wikiarts15"
Get-ChildItem $outRoot -Directory -ErrorAction SilentlyContinue | ForEach-Object {
    $method = $_.Name
    $sumPath = "$($_.FullName)\summary.json"
    $doneMarker = "$($_.FullName)\_DONE"
    $evalDone = if (Test-Path $sumPath) { "EVAL_DONE" } else { "eval_pending" }
    $genDone = if (Test-Path $doneMarker) { "GEN_DONE" } else { "gen_pending" }
    $sumSize = if (Test-Path $sumPath) { (Get-Item $sumPath).Length } else { 0 }
    Write-Output ("{0}: {1}, {2}, summary={3} bytes" -f $method, $genDone, $evalDone, $sumSize)
}

Write-Output ""
Write-Output "=== identity eval.out (tail) ==="
$f = "$repo\logs\baseline_wikiarts15.log.identity.eval.out"
if (Test-Path $f) { Get-Content $f -Tail 20 }

Write-Output ""
Write-Output "=== adain eval.out (tail) ==="
$f = "$repo\logs\baseline_wikiarts15.log.adain.eval.out"
if (Test-Path $f) { Get-Content $f -Tail 20 }

Write-Output ""
Write-Output "=== wct eval.out (tail) ==="
$f = "$repo\logs\baseline_wikiarts15.log.wct.eval.out"
if (Test-Path $f) { Get-Content $f -Tail 20 }

Write-Output ""
Write-Output "=== running python ==="
Get-Process -Name python -ErrorAction SilentlyContinue | Select-Object Id,@{N="WS_MB";E={[math]::Round($_.WorkingSet64/1MB,1)}},StartTime | Format-Table -AutoSize
'@

$bytes = [System.Text.Encoding]::Unicode.GetBytes($script)
$b64 = [Convert]::ToBase64String($bytes)

ssh -p 2222 -o LogLevel=ERROR administrator@100.115.18.62 "powershell -EncodedCommand $b64"
