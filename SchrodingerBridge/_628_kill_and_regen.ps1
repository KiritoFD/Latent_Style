# Kill all python processes on remote, regenerate X configs with full_eval disabled
$ErrorActionPreference = 'Continue'

function Remote-Cmd($cmd) {
    ssh -o ConnectTimeout=10 Administrator@100.115.18.62 -p 2222 $cmd
}

Write-Host "=== Step 1: Stop scheduled tasks ==="
Remote-Cmd 'schtasks /End /TN _628_batch_runner 2>nul'
Remote-Cmd 'schtasks /End /TN _628_watchdog 2>nul'

Write-Host "=== Step 2: Kill all python.exe ==="
Remote-Cmd 'taskkill /F /IM python.exe 2>nul'

Start-Sleep -Seconds 4

Write-Host "=== Step 3: Verify no python.exe running ==="
Remote-Cmd 'tasklist /FI "IMAGENAME eq python.exe" 2>nul | find /c /i "python.exe"'

Write-Host "=== Step 4: Sync updated generator script via scp ==="
scp -P 2222 "G:\GitHub\Latent_Style\SchrodingerBridge\_628_gen_extreme_loss_configs.py" "Administrator@100.115.18.62:C:/Users/Administrator/AppData/Local/Temp/_628_gen_extreme_loss_configs.py"

Write-Host "=== Step 5: Copy to remote project root ==="
Remote-Cmd 'copy /Y "C:\Users\Administrator\AppData\Local\Temp\_628_gen_extreme_loss_configs.py" "I:\Github\Latent_Style\SchrodingerBridge\_628_gen_extreme_loss_configs.py"'

Write-Host "=== Step 6: Verify file size (should be ~9500+ bytes with full_eval fix) ==="
Remote-Cmd 'dir I:\Github\Latent_Style\SchrodingerBridge\_628_gen_extreme_loss_configs.py'

Write-Host "=== Step 7: Run generator on remote ==="
Remote-Cmd 'cd /d I:\Github\Latent_Style\SchrodingerBridge && C:\Progra~1\Python312\python.exe _628_gen_extreme_loss_configs.py'

Write-Host "=== Step 8: Verify X10 config has full_eval disabled ==="
Remote-Cmd 'findstr "full_eval" I:\Github\Latent_Style\SchrodingerBridge\configs\ablations\628_destructive\X10_contrast_w10.json'

Write-Host "=== Done. Ready to restart batch. ==="
