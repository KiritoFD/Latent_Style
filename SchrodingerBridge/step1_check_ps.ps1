# Step 1: Check checkpoints for FC-SB Phase 2 experiments
$ErrorActionPreference = "Continue"

Write-Host "=== Step 1: Check Checkpoints ===" -ForegroundColor Cyan

# Create directory on remote server
Write-Host "`n1. Creating /home/xy directory on remote server..." -ForegroundColor Yellow
& ssh -p 2222 administrator@100.115.18.62 "mkdir -p /home/xy"

# Upload check script
Write-Host "`n2. Uploading check script to remote server..." -ForegroundColor Yellow
& scp -P 2222 "g:\GitHub\Latent_Style\SchrodingerBridge\check_checkpoint.sh" administrator@100.115.18.62:/home/xy/check_checkpoint.sh

# Execute the check script
Write-Host "`n3. Executing checkpoint check on remote server..." -ForegroundColor Yellow
& ssh -p 2222 administrator@100.115.18.62 "chmod +x /home/xy/check_checkpoint.sh && bash /home/xy/check_checkpoint.sh"

# Download results
Write-Host "`n4. Downloading results from remote server..." -ForegroundColor Yellow
& scp -P 2222 administrator@100.115.18.62:/home/xy/checkpoint_check.txt "g:\GitHub\Latent_Style\SchrodingerBridge\checkpoint_check.txt"

# Display results
Write-Host "`n=== Checkpoint Check Results ===" -ForegroundColor Green
if (Test-Path "g:\GitHub\Latent_Style\SchrodingerBridge\checkpoint_check.txt") {
    Get-Content "g:\GitHub\Latent_Style\SchrodingerBridge\checkpoint_check.txt"
} else {
    Write-Host "ERROR: Failed to download results!" -ForegroundColor Red
}
