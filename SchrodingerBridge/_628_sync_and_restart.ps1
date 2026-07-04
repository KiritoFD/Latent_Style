# Sync fixed spectral_losses620.py, delete invalid X10 checkpoint, restart batch
$ErrorActionPreference = 'Continue'

function Remote-Cmd($cmd) {
    ssh -o ConnectTimeout=15 Administrator@100.115.18.62 -p 2222 $cmd
}

Write-Host "=== Step 1: Stop batch and watchdog ==="
Remote-Cmd 'schtasks /End /TN sb_628_batch_runner 2>nul'
Remote-Cmd 'schtasks /End /TN sb_628_watchdog 2>nul'
Remote-Cmd 'taskkill /F /IM python.exe 2>nul'

Start-Sleep -Seconds 4

Write-Host "=== Step 2: Sync fixed spectral_losses620.py (hinge→continuous) ==="
scp -P 2222 "G:\GitHub\Latent_Style\SchrodingerBridge\src\spectral_losses620.py" "Administrator@100.115.18.62:I:/Github/Latent_Style/SchrodingerBridge/src/spectral_losses620.py"

Write-Host "=== Step 3: Verify sync (file size should be ~23K) ==="
Remote-Cmd 'dir I:\Github\Latent_Style\SchrodingerBridge\src\spectral_losses620.py'

Write-Host "=== Step 4: Verify fix is in remote file (search for '628 fix') ==="
Remote-Cmd 'findstr /C:"628 fix" I:\Github\Latent_Style\SchrodingerBridge\src\spectral_losses620.py'

Write-Host "=== Step 5: Delete invalid X10 checkpoint (used old hinge loss, loss=0) ==="
Remote-Cmd 'del /F /Q I:\Github\Latent_Style\SchrodingerBridge\exp\628_ablation\destructive\X10_contrast_w10\epoch_0010.pt 2>nul'
Remote-Cmd 'dir I:\Github\Latent_Style\SchrodingerBridge\exp\628_ablation\destructive\X10_contrast_w10\ 2>nul'

Write-Host "=== Step 6: Restart batch (existing schtasks still configured) ==="
Remote-Cmd 'schtasks /Run /TN sb_628_batch_runner'
Remote-Cmd 'schtasks /Run /TN sb_628_watchdog 2>nul'

Start-Sleep -Seconds 12

Write-Host "=== Step 7: Verify python.exe is running ==="
Remote-Cmd 'tasklist /FI "IMAGENAME eq python.exe"'

Write-Host "=== Step 8: Check batch stdout ==="
Remote-Cmd 'type I:\Github\Latent_Style\SchrodingerBridge\exp\628_ablation\destructive_logs\batch_runner_stdout.log 2>nul'

Write-Host "=== Done. Batch restarted with hinge loss fix. ==="
