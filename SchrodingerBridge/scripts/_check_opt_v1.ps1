$logFile = "I:\Github\Latent_Style\WEAVE\_run_opt_v1.log"
$summaryFile = "I:\Github\Latent_Style\WEAVE\_tmp_opt_v1\summary.json"

Write-Host "=== SUMMARY ==="
if (Test-Path $summaryFile) {
    Write-Host "summary.json EXISTS"
} else {
    Write-Host "no summary yet"
}

Write-Host "=== LOG (tail 25) ==="
if (Test-Path $logFile) {
    Get-Content $logFile -Tail 25
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

Write-Host "=== OUTPUT DIR ==="
$outDir = "I:\Github\Latent_Style\WEAVE\_tmp_opt_v1"
if (Test-Path $outDir) {
    Get-ChildItem $outDir | ForEach-Object { Write-Host $_.Name }
}
