$errLog = "I:\Github\Latent_Style\SchrodingerBridge\exp\p4_fusion_breakout\n1_lvl2_stderr.log"
Write-Host "=== Stderr log tail (last 30 lines) ==="
if (Test-Path $errLog) {
    Get-Content $errLog -Tail 30 | ForEach-Object {
        if ($_.Length -gt 250) {
            Write-Host $_.Substring(0, 250) + "..."
        } else {
            Write-Host $_
        }
    }
} else {
    Write-Host "File not found: $errLog"
}

Write-Host ""
Write-Host "=== Check epoch_0007 eval state ==="
$ep7Dir = "I:\Github\Latent_Style\SchrodingerBridge\exp\p4_fusion_breakout\n1_lvl2_gate03_whh25\full_eval\epoch_0007"
if (Test-Path $ep7Dir) {
    Get-ChildItem $ep7Dir | ForEach-Object { Write-Host $_.Name }
    $summaryFile = "$ep7Dir\summary.json"
    if (Test-Path $summaryFile) {
        Write-Host "summary.json exists"
    } else {
        Write-Host "summary.json MISSING - eval not complete"
    }
} else {
    Write-Host "epoch_0007 dir does not exist"
}

Write-Host ""
Write-Host "=== Check epoch_0008 eval state ==="
$ep8Dir = "I:\Github\Latent_Style\SchrodingerBridge\exp\p4_fusion_breakout\n1_lvl2_gate03_whh25\full_eval\epoch_0008"
if (Test-Path $ep8Dir) {
    Get-ChildItem $ep8Dir | ForEach-Object { Write-Host $_.Name }
} else {
    Write-Host "epoch_0008 dir does not exist"
}

Write-Host ""
Write-Host "=== Check schtasks state ==="
Get-ScheduledTask -TaskName "n1_train" -ErrorAction SilentlyContinue | Select-Object TaskName, State
