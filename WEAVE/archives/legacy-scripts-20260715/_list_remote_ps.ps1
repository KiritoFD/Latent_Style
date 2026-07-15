$cmd = "Get-Process powershell | Select-Object Id,StartTime,@{Name='CmdLine';Expression={`$_.CommandLine}} | Format-Table -AutoSize"
$encoded = [Convert]::ToBase64String([Text.Encoding]::Unicode.GetBytes($cmd))
$expr = "ssh.exe -p 2222 -o LogLevel=ERROR administrator@100.115.18.62 `"powershell -EncodedCommand $encoded`""
Invoke-Expression $expr
