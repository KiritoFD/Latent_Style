# Test SSH connection
$sshPath = "C:\Windows\System32\OpenSSH\ssh.exe"
$args = @("-p", "2222", "administrator@100.115.18.62", "pwd")
Write-Host "Testing SSH connection..."
& $sshPath $args
Write-Host "Exit code: $LASTEXITCODE"
