$cmd = "Get-Process powershell | Where-Object {`$_.CommandLine -like '*pipeline_probe_713_round6*'} | Select-Object Id,StartTime | Format-Table -AutoSize"
$encoded = [Convert]::ToBase64String([Text.Encoding]::Unicode.GetBytes($cmd))
$expr = "ssh.exe -p 2222 -o LogLevel=ERROR administrator@100.115.18.62 `"powershell -EncodedCommand $encoded`""
Write-Host "> $expr"
Invoke-Expression $expr
