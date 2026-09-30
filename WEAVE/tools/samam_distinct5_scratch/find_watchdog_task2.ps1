$ErrorActionPreference = 'SilentlyContinue'

Write-Host "===== All scheduled tasks containing watchdog/628/p7/p8/batch ====="
$tasks = schtasks /Query /FO CSV /V 2>$null | ConvertFrom-Csv
$susp = $tasks | Where-Object {
    $_.TaskName -match 'watchdog|628|p7|p8|destructive|samam' -or
    $_.'Task To Run' -match 'watchdog|628|p7|p8|destructive|samam|_628_'
}
$susp | Select-Object TaskName, Status, 'Next Run Time', 'Last Run Time', 'Last Result', 'Task To Run', 'Schedule Type' | Format-List

Write-Host ""
Write-Host "===== All READY scheduled tasks running powershell/bat/ps1 ====="
$tasks | Where-Object { $_.Status -eq 'Ready' -and $_.'Task To Run' -match 'powershell|\.bat|\.ps1' } |
    Select-Object TaskName, 'Next Run Time', 'Task To Run' | Format-Table -AutoSize -Wrap

Write-Host ""
Write-Host "===== Watchdog log tail (to confirm it ran recently) ====="
$wdlog = "I:\Github\Latent_Style\SchrodingerBridge\exp\628_ablation\destructive_logs\watchdog.log"
if (Test-Path $wdlog) { Get-Content $wdlog -Tail 30 } else { Write-Host "(no watchdog.log)" }

Write-Host ""
Write-Host "===== batch_runner.pid file ====="
$pidf = "I:\Github\Latent_Style\SchrodingerBridge\exp\628_ablation\destructive_logs\batch_runner.pid"
if (Test-Path $pidf) { "PID file content: $(Get-Content $pidf -Raw)" } else { Write-Host "(no pid file)" }
