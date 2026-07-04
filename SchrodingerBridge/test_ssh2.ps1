# Test SSH connection with proper argument handling
$sshPath = "C:\Windows\System32\OpenSSH\ssh.exe"
$remoteCmd = 'pwd'
Write-Host "Testing SSH connection..."
Start-Process -FilePath $sshPath -ArgumentList "-p 2222 administrator@100.115.18.62 $remoteCmd" -Wait -NoNewWindow
Write-Host "Exit code: $LASTEXITCODE"
