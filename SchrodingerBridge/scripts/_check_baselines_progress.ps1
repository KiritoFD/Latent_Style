# Monitor baselines progress
$script = @'
$repo = "I:\Github\Latent_Style\SchrodingerBridge"
$logMain = "$repo\logs\baseline_wikiarts15.log"

Write-Output "=== main log (last 30 lines) ==="
if (Test-Path $logMain) {
    Get-Content $logMain -Tail 30
} else {
    Write-Output "log not yet created"
}

Write-Output ""
Write-Output "=== per-method log files ==="
Get-ChildItem "$repo\logs\baseline_wikiarts15*" -ErrorAction SilentlyContinue | ForEach-Object {
    Write-Output ("{0}`t{1} bytes`t{2}" -f $_.Name, $_.Length, $_.LastWriteTime)
}

Write-Output ""
Write-Output "=== output dir structure ==="
$outRoot = "$repo\exp\baseline_wikiarts15"
if (Test-Path $outRoot) {
    Get-ChildItem $outRoot -ErrorAction SilentlyContinue | ForEach-Object {
        $doneMarker = "$($_.FullName)\_DONE"
        $imgCount = 0
        $imgDir = "$($_.FullName)\images"
        if (Test-Path $imgDir) {
            $imgCount = (Get-ChildItem $imgDir -File -ErrorAction SilentlyContinue | Measure-Object).Count
        }
        $done = if (Test-Path $doneMarker) { "DONE" } else { "..." }
        Write-Output ("{0}`t{1}`timages={2}" -f $_.Name, $done, $imgCount)
    }
} else {
    Write-Output "output dir not yet created"
}

Write-Output ""
Write-Output "=== running python processes ==="
Get-Process -Name python -ErrorAction SilentlyContinue | Select-Object Id,ProcessName,@{N="WS_MB";E={[math]::Round($_.WorkingSet64/1MB,1)}},StartTime | Format-Table -AutoSize
'@

$bytes = [System.Text.Encoding]::Unicode.GetBytes($script)
$b64 = [Convert]::ToBase64String($bytes)

ssh -p 2222 -o LogLevel=ERROR administrator@100.115.18.62 "powershell -EncodedCommand $b64"
