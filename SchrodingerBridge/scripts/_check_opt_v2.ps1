$logFile = "I:\Github\Latent_Style\WEAVE\_run_opt_v2.log"
$summaryFile = "I:\Github\Latent_Style\WEAVE\_tmp_opt_v1\summary.json"
$outDir = "I:\Github\Latent_Style\WEAVE\_tmp_opt_v1"

Write-Host "=== SUMMARY ==="
if (Test-Path $summaryFile) {
    Write-Host "summary.json EXISTS"
} else {
    Write-Host "no summary yet"
}

Write-Host "=== LOG size ==="
if (Test-Path $logFile) {
    $size = (Get-Item $logFile).Length
    Write-Host "log size: $size bytes"
    Write-Host "=== LOG (last 40) ==="
    Get-Content $logFile -Tail 40
} else {
    Write-Host "no log"
}

Write-Host "=== PYTHON PROC ==="
$procs = Get-Process python -ErrorAction SilentlyContinue
if ($procs) {
    foreach ($p in $procs) {
        Write-Host ("PID=" + $p.Id + " CPU=" + $p.CPU + " WS=" + [math]::Round($p.WorkingSet/1MB) + "MB")
    }
} else {
    Write-Host "no python running"
}

Write-Host "=== IMAGES DIR ==="
$imagesDir = Join-Path $outDir "images"
if (Test-Path $imagesDir) {
    $count = (Get-ChildItem $imagesDir -File).Count
    Write-Host "images count: $count"
} else {
    Write-Host "no images dir"
}

Write-Host "=== OUTPUT DIR FILES ==="
if (Test-Path $outDir) {
    Get-ChildItem $outDir -File | ForEach-Object { Write-Host ($_.Name + " (" + $_.Length + " bytes)") }
}
