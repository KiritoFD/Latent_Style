# Test basic SSH connection without command
$sshPath = "C:\Windows\System32\OpenSSH\ssh.exe"
Write-Host "Test 1: SSH version"
& $sshPath -V
Write-Host ""
Write-Host "Test 2: Try to connect (will prompt for password or fail)"
& $sshPath -p 2222 -o BatchMode=yes -o ConnectTimeout=5 administrator@100.115.18.62 echo "connected" 2>&1
