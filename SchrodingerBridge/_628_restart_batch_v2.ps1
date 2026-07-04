# Restart batch runner on remote via SSH (decoupled from SSH session via schtasks)
$ErrorActionPreference = 'Continue'

function Remote-Cmd($cmd) {
    ssh -o ConnectTimeout=15 Administrator@100.115.18.62 -p 2222 $cmd
}

Write-Host "=== Step 1: Delete old schtasks ==="
Remote-Cmd 'schtasks /Delete /TN sb_628_batch_runner /F 2>nul'
Remote-Cmd 'schtasks /Delete /TN sb_628_watchdog /F 2>nul'

Write-Host "=== Step 2: Write .bat launcher ==="
$batContent = @'
@echo off
set PYTHON=C:\Progra~1\Python312\python.exe
set RUNNER=I:\Github\Latent_Style\SchrodingerBridge\628_run_destructive_batch.py
set STDOUT=I:\Github\Latent_Style\SchrodingerBridge\exp\628_ablation\destructive_logs\batch_runner_stdout.log
set STDERR=I:\Github\Latent_Style\SchrodingerBridge\exp\628_ablation\destructive_logs\batch_runner_stderr.log
cd /d I:\Github\Latent_Style\SchrodingerBridge
"%PYTHON%" "%RUNNER%" > "%STDOUT%" 2> "%STDERR%"
'@
$tmpBat = "G:\GitHub\Latent_Style\SchrodingerBridge\_628_batch_runner.bat"
[System.IO.File]::WriteAllText($tmpBat, $batContent, [System.Text.Encoding]::Default)
scp -P 2222 $tmpBat "Administrator@100.115.18.62:I:/Github/Latent_Style/SchrodingerBridge/_628_batch_runner.bat"

Write-Host "=== Step 3: Create and run batch schtask ==="
Remote-Cmd 'schtasks /Create /TN sb_628_batch_runner /TR "I:\Github\Latent_Style\SchrodingerBridge\_628_batch_runner.bat" /SC ONCE /ST 23:59 /RU Administrator /IT /F'
Remote-Cmd 'schtasks /Run /TN sb_628_batch_runner'

Start-Sleep -Seconds 12

Write-Host "=== Step 4: Verify python.exe is running ==="
Remote-Cmd 'tasklist /FI "IMAGENAME eq python.exe" 2>nul'

Write-Host "=== Step 5: Check batch runner stdout (first 20 lines) ==="
Remote-Cmd 'type I:\Github\Latent_Style\SchrodingerBridge\exp\628_ablation\destructive_logs\batch_runner_stdout.log 2>nul'

Write-Host "=== Step 6: Start watchdog (every 5 min) ==="
Remote-Cmd 'schtasks /Create /TN sb_628_watchdog /TR "powershell.exe -ExecutionPolicy Bypass -NoProfile -File I:\Github\Latent_Style\SchrodingerBridge\_628_watchdog.ps1" /SC MINUTE /MO 5 /RU Administrator /IT /F'
Remote-Cmd 'schtasks /Run /TN sb_628_watchdog'

Write-Host "=== Step 7: Count progress ==="
Remote-Cmd 'powershell -NoProfile -Command "$cfgs = Get-ChildItem I:\Github\Latent_Style\SchrodingerBridge\configs\ablations\628_destructive\*.json; $done = 0; $pending = 0; foreach ($c in $cfgs) { $ep10 = \"I:\Github\Latent_Style\SchrodingerBridge\exp\628_ablation\destructive\$($c.BaseName)\epoch_0010.pt\"; if (Test-Path $ep10) { $done++ } else { $pending++ } }; Write-Host \"Total: $($cfgs.Count) Done: $done Pending: $pending\""'

Write-Host "=== Batch restarted with full_eval disabled. ==="
