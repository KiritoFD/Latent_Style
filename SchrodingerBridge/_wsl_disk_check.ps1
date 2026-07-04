# Check WSL vhdx file size and disk usage
Write-Host "=== WSL vhdx files ===" -ForegroundColor Cyan
Get-ChildItem -Path "$env:LOCALAPPDATA\Packages" -Recurse -Filter "ext4.vhdx" -ErrorAction SilentlyContinue | ForEach-Object {
    $sizeGB = [math]::Round($_.Length / 1GB, 2)
    Write-Host "Path: $($_.FullName)"
    Write-Host "Size: $sizeGB GB"
    Write-Host "LastWrite: $($_.LastWriteTime)"
    Write-Host "---"
}

Write-Host "`n=== Disk C: usage ===" -ForegroundColor Cyan
Get-PSDrive C | Format-Table Name, @{N='UsedGB';E={[math]::Round($_.Used/1GB,2)}}, @{N='FreeGB';E={[math]::Round($_.Free/1GB,2)}}, @{N='TotalGB';E={[math]::Round(($_.Used+$_.Free)/1GB,2)}}
