# Debug why batch runner PID=26752 exited
$taskName = 'sb_628_batch_runner'

Write-Host "=== schtasks query ==="
schtasks /Query /TN $taskName /V /FO LIST 2>&1 | Select-String -Pattern 'Last Result|Last Run Time|Next Run Time|Status|Task To Run|Run As User|Schedule Type'

Write-Host "`n=== batch_runner_stderr.log ==="
$stderrLog = 'I:/Github/Latent_Style/SchrodingerBridge/exp/628_ablation/destructive_logs/batch_runner_stderr.log'
if (Test-Path $stderrLog) {
    $size = (Get-Item $stderrLog).Length
    Write-Host "Size: $size bytes"
    if ($size -gt 0) {
        Get-Content $stderrLog -Tail 30
    } else {
        Write-Host "(empty)"
    }
}

Write-Host "`n=== batch_runner_stdout.log tail ==="
$stdoutLog = 'I:/Github/Latent_Style/SchrodingerBridge/exp/628_ablation/destructive_logs/batch_runner_stdout.log'
if (Test-Path $stdoutLog) {
    Get-Content $stdoutLog -Tail 15
}

Write-Host "`n=== All python processes ==="
Get-Process python -ErrorAction SilentlyContinue | Format-Table Id,StartTime,CPU,@{N='WS_MB';E={[math]::Round($_.WorkingSet64/1MB,1)}}

Write-Host "`n=== D20 training log tail ==="
$d20Log = 'I:/Github/Latent_Style/SchrodingerBridge/exp/628_ablation/destructive_logs/D20_attn_relu2.log'
if (Test-Path $d20Log) {
    Get-Content $d20Log -Tail 15
}

Write-Host "`n=== System event log (last 10 min, application errors) ==="
try {
    $events = Get-WinEvent -FilterHashtable @{LogName='Application'; Level=2; StartTime=(Get-Date).AddMinutes(-30)} -MaxEvents 5 -ErrorAction SilentlyContinue
    if ($events) {
        foreach ($e in $events) {
            Write-Host "  [$($e.TimeCreated)] $($e.ProviderName): $($e.Message.Substring(0, [Math]::Min(150, $e.Message.Length)))"
        }
    } else {
        Write-Host "  (no recent application errors)"
    }
} catch {
    Write-Host "  (cannot read event log)"
}
