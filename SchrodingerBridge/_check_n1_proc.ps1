Write-Host "=== tasklist python ==="
& tasklist /FI "IMAGENAME eq python.exe" 2>&1

Write-Host ""
Write-Host "=== tasklist powershell ==="
& tasklist /FI "IMAGENAME eq powershell.exe" 2>&1

Write-Host ""
Write-Host "=== Get-ScheduledTask n1_train ==="
$task = Get-ScheduledTask -TaskName "n1_train" -ErrorAction SilentlyContinue
if ($task) {
    $task | Select-Object TaskName, State
    $info = Get-ScheduledTaskInfo -TaskName "n1_train"
    $info | Select-Object LastRunTime, LastTaskResult, NextRunTime, NumberOfMissedRuns
}

Write-Host ""
Write-Host "=== Stderr tail ==="
$errLog = "I:\Github\Latent_Style\SchrodingerBridge\exp\p4_fusion_breakout\n1_lvl2_stderr.log"
if (Test-Path $errLog) {
    Get-Content $errLog -Tail 5 | ForEach-Object {
        if ($_.Length -gt 300) {
            Write-Host $_.Substring(0, 300) + "..."
        } else {
            Write-Host $_
        }
    }
}

Write-Host ""
Write-Host "=== Stdout tail ==="
$outLog = "I:\Github\Latent_Style\SchrodingerBridge\exp\p4_fusion_breakout\n1_lvl2_stdout.log"
if (Test-Path $outLog) {
    Get-Content $outLog -Tail 10 | ForEach-Object {
        if ($_.Length -gt 300) {
            Write-Host $_.Substring(0, 300) + "..."
        } else {
            Write-Host $_
        }
    }
} else {
    Write-Host "stdout log not found"
}

Write-Host ""
Write-Host "=== epoch_0007 summary check ==="
$ep7Summary = "I:\Github\Latent_Style\SchrodingerBridge\exp\p4_fusion_breakout\n1_lvl2_gate03_whh25\full_eval\epoch_0007\summary.json"
if (Test-Path $ep7Summary) {
    Write-Host "summary.json EXISTS - epoch 7 eval complete"
} else {
    Write-Host "summary.json MISSING - epoch 7 eval not complete"
}

Write-Host ""
Write-Host "=== epoch_0008 dir check ==="
$ep8Dir = "I:\Github\Latent_Style\SchrodingerBridge\exp\p4_fusion_breakout\n1_lvl2_gate03_whh25\full_eval\epoch_0008"
if (Test-Path $ep8Dir) {
    Write-Host "epoch_0008 dir exists"
    Get-ChildItem $ep8Dir | ForEach-Object { Write-Host $_.Name }
} else {
    Write-Host "epoch_0008 dir does not exist"
}
