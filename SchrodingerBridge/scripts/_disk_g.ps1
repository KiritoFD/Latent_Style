$d = Get-PSDrive G
$freeGB = [math]::Round($d.Free / 1GB, 2)
$usedGB = [math]::Round($d.Used / 1GB, 2)
Write-Host "G: drive - Used: $usedGB GB, Free: $freeGB GB"
