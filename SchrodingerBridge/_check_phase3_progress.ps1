$logFile = "I:\Github\Latent_Style\SchrodingerBridge\exp\p4_fusion_breakout\phase3_stdout.log"
Write-Host "=== Phase 3 stdout tail (last 50 lines) ==="
if (Test-Path $logFile) {
    Get-Content $logFile -Tail 50 | ForEach-Object {
        if ($_.Length -gt 250) {
            Write-Host $_.Substring(0, 250) + "..."
        } else {
            Write-Host $_
        }
    }
} else {
    Write-Host "log not found"
}

Write-Host ""
Write-Host "=== Phase 3 stderr tail (last 20 lines) ==="
$errFile = "I:\Github\Latent_Style\SchrodingerBridge\exp\p4_fusion_breakout\phase3_stderr.log"
if (Test-Path $errFile) {
    Get-Content $errFile -Tail 20 | ForEach-Object {
        if ($_.Length -gt 250) {
            Write-Host $_.Substring(0, 250) + "..."
        } else {
            Write-Host $_
        }
    }
} else {
    Write-Host "stderr not found"
}

Write-Host ""
Write-Host "=== Phase 3 generated result files ==="
$dir = "I:\Github\Latent_Style\SchrodingerBridge\exp\p4_fusion_breakout\infer_ablation"
Get-ChildItem $dir -Filter "P3_*.json" | Where-Object { $_.Name -notmatch "_override" } | Sort-Object LastWriteTime -Descending | Select-Object -First 10 | ForEach-Object {
    Write-Host "$($_.LastWriteTime) - $($_.Name)"
}

Write-Host ""
Write-Host "=== tasklist python ==="
& tasklist /FI "IMAGENAME eq python.exe" 2>&1 | Select-Object -First 10

Write-Host ""
Write-Host "=== Get-ScheduledTask phase3_abl ==="
Get-ScheduledTask -TaskName "phase3_abl" -ErrorAction SilentlyContinue | Select-Object TaskName, State
