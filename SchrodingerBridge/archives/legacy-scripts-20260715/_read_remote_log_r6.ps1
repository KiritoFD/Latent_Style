$path = $args[0]
$cmd = "if (Test-Path '$path') { Get-Content '$path' -Tail 20 } else { Write-Host 'FILE_NOT_FOUND: $path' }"
$encoded = [Convert]::ToBase64String([Text.Encoding]::Unicode.GetBytes($cmd))
$expr = "ssh.exe -p 2222 -o LogLevel=ERROR administrator@100.115.18.62 `"powershell -EncodedCommand $encoded`""
Invoke-Expression $expr
