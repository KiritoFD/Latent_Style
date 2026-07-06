# Check WD-VF eval status
"=== wikiarts20_all.log FULL ==="
$allLog = "I:\Github\Latent_Style\SchrodingerBridge\logs\wikiarts20_all.log"
if (Test-Path $allLog) { Get-Content $allLog } else { "NOT FOUND" }

"=== wikiarts20_eval.log ==="
$evalLog = "I:\Github\Latent_Style\SchrodingerBridge\logs\wikiarts20_eval.log"
if (Test-Path $evalLog) { Get-Content $evalLog -Tail 20 } else { "NOT FOUND" }

"=== wikiarts20_eval.log.out ==="
$outLog = "I:\Github\Latent_Style\SchrodingerBridge\logs\wikiarts20_eval.log.out"
if (Test-Path $outLog) {
    "size=$((Get-Item $outLog).Length)"
    Get-Content $outLog -Tail 15
} else { "NOT FOUND" }

"=== wikiarts20_eval.log.err ==="
$errLog = "I:\Github\Latent_Style\SchrodingerBridge\logs\wikiarts20_eval.log.err"
if (Test-Path $errLog) {
    "size=$((Get-Item $errLog).Length)"
    Get-Content $errLog -Tail 15
} else { "NOT FOUND" }

"=== WD-VF output dir ==="
$outDir = "I:\Github\Latent_Style\SchrodingerBridge\exp\wikiarts20_eval"
if (Test-Path $outDir) {
    Get-ChildItem $outDir | Select-Object Name, Length, LastWriteTime | Format-Table -AutoSize
} else { "NOT FOUND" }

"=== schtasks status ==="
schtasks /Query /TN wikiarts20_all /V /FO CSV 2>$null | ConvertFrom-Csv | ForEach-Object { $_.PSObject.Properties } | ForEach-Object { if ($_.Value -match 'running|running|267009|0x') { "$($_.Name) = $($_.Value)" } }

"=== processes ==="
Get-Process python,powershell -ErrorAction SilentlyContinue | Select-Object Id, ProcessName, StartTime, CPU | Format-Table -AutoSize
