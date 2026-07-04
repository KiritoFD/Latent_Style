# FC-SB Phase 2 - Check Remote Directory Structure
$sshHost = "administrator@100.115.18.62"
$sshPort = "2222"

Write-Host "Checking remote directory structure..."

# List home directory
Write-Host "`n=== /home/xy/ contents ==="
& ssh -p $sshPort $sshHost 'ls -la /home/xy/' 2>&1

# Check if Latent_Style exists
Write-Host "`n=== Checking /home/xy/Latent_Style ==="
& ssh -p $sshPort $sshHost 'ls -la /home/xy/Latent_Style/ 2>&1 || echo Directory not found'

# Check if SchrodingerBridge exists anywhere
Write-Host "`n=== Searching for SchrodingerBridge ==="
& ssh -p $sshPort $sshHost 'find /home/xy -name "SchrodingerBridge" -type d 2>/dev/null | head -5'
