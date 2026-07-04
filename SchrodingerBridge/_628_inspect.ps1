$root = 'I:\Github\Latent_Style\SchrodingerBridge'

Write-Host "=== batch_runner_stderr.log (last 30) ==="
$errLog = "$root\exp\628_ablation\destructive_logs\batch_runner_stderr.log"
if (Test-Path $errLog) {
    Get-Content $errLog -Tail 30
} else {
    Write-Host "not found"
}

Write-Host ""
Write-Host "=== batch_runner_stdout.log (last 30) ==="
$outLog = "$root\exp\628_ablation\destructive_logs\batch_runner_stdout.log"
if (Test-Path $outLog) {
    Get-Content $outLog -Tail 30
} else {
    Write-Host "not found"
}

Write-Host ""
Write-Host "=== Full batch_log.txt (last 50) ==="
$batchLog = "$root\exp\628_ablation\destructive_logs\batch_log.txt"
if (Test-Path $batchLog) {
    Get-Content $batchLog -Tail 50
}

Write-Host ""
Write-Host "=== All python processes with command line ==="
Get-WmiObject Win32_Process -Filter "Name='python.exe'" | Select-Object ProcessId, CreationDate, CommandLine | Format-List

Write-Host ""
Write-Host "=== All cmd.exe processes with command line ==="
Get-WmiObject Win32_Process -Filter "Name='cmd.exe'" | Select-Object ProcessId, CreationDate, CommandLine | Format-List

Write-Host ""
Write-Host "=== schtasks status ==="
schtasks /Query /TN 'sb_628_batch_runner' 2>&1
schtasks /Query /TN 'sb_628_watchdog' 2>&1
