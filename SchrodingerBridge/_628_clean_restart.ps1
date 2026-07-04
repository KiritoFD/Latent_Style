# Sync updated spectral_losses620.py (periodic debug), clean X10/X11 checkpoints, restart
$ErrorActionPreference = 'Continue'

function Remote-Cmd($cmd) {
    ssh -o ConnectTimeout=15 Administrator@100.115.18.62 -p 2222 $cmd
}

Write-Host "=== Step 1: Stop batch and kill python ==="
Remote-Cmd 'schtasks /End /TN sb_628_batch_runner 2>nul'
Remote-Cmd 'schtasks /End /TN sb_628_watchdog 2>nul'
Remote-Cmd 'taskkill /F /IM python.exe 2>nul'

Start-Sleep -Seconds 4

Write-Host "=== Step 2: Sync updated spectral_losses620.py (periodic debug every 100 steps) ==="
scp -P 2222 "G:\GitHub\Latent_Style\SchrodingerBridge\src\spectral_losses620.py" "Administrator@100.115.18.62:I:/Github/Latent_Style/SchrodingerBridge/src/spectral_losses620.py"

Write-Host "=== Step 3: Verify sync ==="
Remote-Cmd 'dir I:\Github\Latent_Style\SchrodingerBridge\src\spectral_losses620.py'

Write-Host "=== Step 4: Delete X10 ALL checkpoints (only trained 1 epoch with old code) ==="
Remote-Cmd 'del /F /Q I:\Github\Latent_Style\SchrodingerBridge\exp\628_ablation\destructive\X10_contrast_w10\epoch_*.pt 2>nul'
Remote-Cmd 'dir I:\Github\Latent_Style\SchrodingerBridge\exp\628_ablation\destructive\X10_contrast_w10\*.pt 2>nul'

Write-Host "=== Step 5: Delete X11 ALL checkpoints (trained with old hinge loss) ==="
Remote-Cmd 'del /F /Q I:\Github\Latent_Style\SchrodingerBridge\exp\628_ablation\destructive\X11_contrast_w50\epoch_*.pt 2>nul'
Remote-Cmd 'dir I:\Github\Latent_Style\SchrodingerBridge\exp\628_ablation\destructive\X11_contrast_w50\*.pt 2>nul'

Write-Host "=== Step 6: Restart batch ==="
Remote-Cmd 'schtasks /Run /TN sb_628_batch_runner'
Remote-Cmd 'schtasks /Run /TN sb_628_watchdog 2>nul'

Start-Sleep -Seconds 15

Write-Host "=== Step 7: Check batch stdout ==="
Remote-Cmd 'type I:\Github\Latent_Style\SchrodingerBridge\exp\628_ablation\destructive_logs\batch_runner_stdout.log 2>nul'

Write-Host "=== Step 8: Check X10 log (should show periodic 628-ALL-DEBUG) ==="
Remote-Cmd 'powershell -NoProfile -Command "if (Test-Path I:\Github\Latent_Style\SchrodingerBridge\exp\628_ablation\destructive_logs\X10_contrast_w10.log) { Get-Content I:\Github\Latent_Style\SchrodingerBridge\exp\628_ablation\destructive_logs\X10_contrast_w10.log -Tail 15 } else { Write-Host \"X10 log not yet created\" }"'

Write-Host "=== Done. Batch restarted with periodic debug. ==="
