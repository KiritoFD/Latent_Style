# Test SSH with escaped arguments
$sshPath = "C:\Windows\System32\OpenSSH\ssh.exe"
Write-Host "Testing SSH with different approaches..."
Write-Host ""

# Approach 1: Using --% to stop parsing
Write-Host "Approach 1: Using --% operator"
& $sshPath --% -p 2222 administrator@100.115.18.62 pwd
Write-Host "Exit code: $LASTEXITCODE"
Write-Host ""
