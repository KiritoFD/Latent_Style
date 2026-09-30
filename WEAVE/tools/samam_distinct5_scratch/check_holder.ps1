Write-Host "=== WSL.EXE PROCESSES ==="
Get-Process -Name wsl -ErrorAction SilentlyContinue | Format-Table Id, ProcessName, StartTime -AutoSize
Write-Host ""
Write-Host "=== WSL STATE ==="
&wsl -l -v
Write-Host ""
Write-Host "=== HOLDER LOG ==="
if (Test-Path C:\Users\Administrator\wsl_holder.log) { Get-Content C:\Users\Administrator\wsl_holder.log -Tail 10 }
Write-Host ""
Write-Host "=== HOLDER TASK STATUS ==="
schtasks /Query /TN WSL_Holder /FO LIST 2>$null | Select-String "Status"
Write-Host ""
Write-Host "=== WSL INTERNAL (via script) ==="
&wsl -d Ubuntu-22.04 -e bash /mnt/c/Users/Administrator/diag_remote.sh 2>&1 | Select-Object -First 30
