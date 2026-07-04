# 628 fix: sync code, delete X checkpoints (buggy), restart batch
$ErrorActionPreference = 'Continue'
$remote = 'Administrator@100.115.18.62'
$port = '2222'
$remoteRoot = 'I:\Github\Latent_Style\SchrodingerBridge'
$localRoot = 'g:\GitHub\Latent_Style\SchrodingerBridge'

function Remote-Cmd($cmd) {
    ssh -o ConnectTimeout=15 $remote -p $port $cmd
}

# Step 1: Kill all python processes on remote
Write-Host "=== Step 1: Killing python processes ==="
Remote-Cmd "taskkill /F /IM python.exe"
Start-Sleep -Seconds 3

# Step 2: Stop batch runner and watchdog schtasks
Write-Host "=== Step 2: Stopping schtasks ==="
Remote-Cmd "schtasks /End /TN sb_628_batch_runner 2>nul"
Remote-Cmd "schtasks /End /TN sb_628_watchdog 2>nul"
Start-Sleep -Seconds 2

# Step 3: Sync fixed spectral_losses620.py
Write-Host "=== Step 3: Syncing spectral_losses620.py ==="
scp -P $port "$localRoot\src\spectral_losses620.py" "${remote}:$remoteRoot\src\spectral_losses620.py"
if ($LASTEXITCODE -ne 0) {
    Write-Host "ERROR: scp failed for spectral_losses620.py"
    exit 1
}
Write-Host "Synced spectral_losses620.py"

# Step 4: Delete all X experiment checkpoints (buggy code)
Write-Host "=== Step 4: Deleting X experiment checkpoints ==="
$xExps = @('X1_velmag_w10','X2_velmag_w50','X3_velmag_w100','X4_dir_cos_w10','X5_dir_cos_w50','X6_dir_cos_w100','X7_outvar_w10','X8_outvar_w50','X9_outvar_w100','X10_contrast_w10','X11_contrast_w50','X12_contrast_w100','X13_chvar_w10','X14_chvar_w50','X15_chvar_w100','X16_hfenergy_w10','X17_hfenergy_w50','X18_hfenergy_w100','X19_colormatch_w10','X20_colormatch_w50','X21_colormatch_w100','X22_hsvsat_w1','X23_hsvsat_w10','X24_hsvsat_w50','X25_attnent_w1','X26_attnent_w10','X27_attnent_w50','X28_combo_content_w50','X29_combo_direction_w50','X30_combo_all_w10','X31_combo_all_w50')
foreach ($x in $xExps) {
    $delCmd = "if exist `"$remoteRoot\exp\628_ablation\destructive\$x`" rmdir /S /Q `"$remoteRoot\exp\628_ablation\destructive\$x`""
    Remote-Cmd $delCmd
    Write-Host "  Deleted $x"
}

# Step 5: Clean batch_runner_stdout.log
Write-Host "=== Step 5: Cleaning batch runner logs ==="
Remote-Cmd "if exist `"$remoteRoot\exp\628_ablation\destructive_logs\batch_runner_stdout.log`" del /Q `"$remoteRoot\exp\628_ablation\destructive_logs\batch_runner_stdout.log`""

# Step 6: Verify code sync
Write-Host "=== Step 6: Verifying code sync ==="
$verify = Remote-Cmd "findstr /C:`"628 fix`" /C:`"squared relative difference`" `"$remoteRoot\src\spectral_losses620.py`""
Write-Host $verify

# Step 7: Restart batch runner via schtasks
Write-Host "=== Step 7: Restarting batch runner ==="
Remote-Cmd "schtasks /Run /TN sb_628_batch_runner"
Start-Sleep -Seconds 5

# Step 8: Restart watchdog
Write-Host "=== Step 8: Restarting watchdog ==="
Remote-Cmd "schtasks /Run /TN sb_628_watchdog"

Write-Host ""
Write-Host "=== DONE: Batch restarted with fixed code ==="
