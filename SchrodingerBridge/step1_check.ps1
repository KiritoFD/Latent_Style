# Checkpoint inspection script for FC-SB Phase 2 experiments
$sshCommand = 'ssh -p 2222 administrator@100.115.18.62'

# Create directory if needed
Invoke-Expression "$sshCommand `"mkdir -p /home/xy`""

# Upload check script
$localScript = "g:\GitHub\Latent_Style\SchrodingerBridge\check_checkpoint.sh"
$remotePath = "/home/xy/check_checkpoint.sh"
Write-Host "Uploading check script..."
scp -P 2222 $localScript "administrator@100.115.18.62:$remotePath"

# Execute the check script
Write-Host "`nExecuting checkpoint check on remote server..."
Invoke-Expression "$sshCommand `"chmod +x /home/xy/check_checkpoint.sh && bash /home/xy/check_checkpoint.sh`""

# Download results
Write-Host "`nDownloading results..."
scp -P 2222 "administrator@100.115.18.62:/home/xy/checkpoint_check.txt" "g:\GitHub\Latent_Style\SchrodingerBridge\checkpoint_check.txt"

Write-Host "`nDone! Results saved to checkpoint_check.txt"
