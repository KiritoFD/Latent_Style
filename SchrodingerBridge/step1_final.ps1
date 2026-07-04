# Step 1: Check checkpoints for FC-SB Phase 2 experiments
$sshPath = "C:\Windows\System32\OpenSSH\ssh.exe"
$scpPath = "C:\Windows\System32\OpenSSH\scp.exe"
$remoteHost = "administrator@100.115.18.62"
$port = "2222"

Write-Host "=== FC-SB Phase 2: Checkpoint Inspection ===" -ForegroundColor Cyan
Write-Host ""

# Step 1: Create directory on remote server
Write-Host "[1/4] Creating /home/xy directory..." -ForegroundColor Yellow
& $sshPath -p $port $remoteHost "mkdir -p /home/xy"
Write-Host "Done." -ForegroundColor Green

# Step 2: Upload check script
Write-Host "`n[2/4] Uploading check script..." -ForegroundColor Yellow
& $scpPath -P $port "g:\GitHub\Latent_Style\SchrodingerBridge\check_checkpoint.sh" "${remoteHost}:/home/xy/check_checkpoint.sh"
Write-Host "Done." -ForegroundColor Green

# Step 3: Execute check script
Write-Host "`n[3/4] Executing checkpoint inspection..." -ForegroundColor Yellow
& $sshPath -p $port $remoteHost "chmod +x /home/xy/check_checkpoint.sh && bash /home/xy/check_checkpoint.sh"
Write-Host "Done." -ForegroundColor Green

# Step 4: Download results
Write-Host "`n[4/4] Downloading results..." -ForegroundColor Yellow
& $scpPath -P $port "${remoteHost}:/home/xy/checkpoint_check.txt" "g:\GitHub\Latent_Style\SchrodingerBridge\checkpoint_check.txt"
Write-Host "Done." -ForegroundColor Green

# Display results
Write-Host "`n" + "="*60 -ForegroundColor Cyan
Write-Host "CHECKPOINT INSPECTION RESULTS:" -ForegroundColor White
Write-Host "="*60 -ForegroundColor Cyan
Get-Content "g:\GitHub\Latent_Style\SchrodingerBridge\checkpoint_check.txt"
