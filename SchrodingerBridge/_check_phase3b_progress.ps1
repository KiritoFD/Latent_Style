$logFile = "I:\Github\Latent_Style\SchrodingerBridge\exp\p4_fusion_breakout\phase3b_stdout.log"
Write-Host "=== Phase 3b stdout tail (last 80 lines) ==="
if (Test-Path $logFile) {
    Get-Content $logFile -Tail 80 | ForEach-Object {
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
Write-Host "=== Phase 3b generated result files ==="
$dir = "I:\Github\Latent_Style\SchrodingerBridge\exp\p4_fusion_breakout\infer_ablation"
Get-ChildItem $dir -Filter "P3b_*.json" | Where-Object { $_.Name -notmatch "_override" } | Sort-Object LastWriteTime -Descending | Select-Object -First 15 | ForEach-Object {
    Write-Host "$($_.LastWriteTime) - $($_.Name)"
}

Write-Host ""
Write-Host "=== tasklist python ==="
& tasklist /FI "IMAGENAME eq python.exe" 2>&1 | Select-Object -First 10

Write-Host ""
Write-Host "=== Get-ScheduledTask phase3b_abl ==="
Get-ScheduledTask -TaskName "phase3b_abl" -ErrorAction SilentlyContinue | Select-Object TaskName, State
