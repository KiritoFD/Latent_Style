# Find all vhdx files on C: drive
Write-Host "=== Searching for *.vhdx files on C: ===" -ForegroundColor Cyan
Get-ChildItem -Path C:\ -Recurse -Filter "*.vhdx" -ErrorAction SilentlyContinue | ForEach-Object {
    $sizeGB = [math]::Round($_.Length / 1GB, 2)
    Write-Host "Path: $($_.FullName)"
    Write-Host "Size: $sizeGB GB"
    Write-Host "LastWrite: $($_.LastWriteTime)"
    Write-Host "---"
}

Write-Host "`n=== WSL distro registry ===" -ForegroundColor Cyan
Get-ChildItem -Path "HKCU:\Software\Microsoft\Windows\CurrentVersion\Lxss" -ErrorAction SilentlyContinue | ForEach-Object {
    $props = Get-ItemProperty $_.PSPath
    Write-Host "DistroName: $($props.DistributionName)"
    Write-Host "BasePath: $($props.BasePath)"
    Write-Host "---"
}
