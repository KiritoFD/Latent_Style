$ErrorActionPreference = 'SilentlyContinue'
Set-Location 'g:\GitHub\Latent_Style\SchrodingerBridge'
Write-Host "=== Top-level directory sizes ==="
Get-ChildItem -Directory | ForEach-Object {
    $size = (Get-ChildItem $_.FullName -Recurse -ErrorAction SilentlyContinue | Measure-Object -Property Length -Sum).Sum
    [PSCustomObject]@{
        Name = $_.Name
        SizeMB = [math]::Round($size/1MB, 1)
        LastWrite = $_.LastWriteTime.ToString('yyyy-MM-dd')
    }
} | Sort-Object SizeMB -Descending | Format-Table -AutoSize

Write-Host "`n=== exp/ subdirectory sizes ==="
Get-ChildItem -Path 'exp' -Directory | ForEach-Object {
    $size = (Get-ChildItem $_.FullName -Recurse -ErrorAction SilentlyContinue | Measure-Object -Property Length -Sum).Sum
    [PSCustomObject]@{
        Name = $_.Name
        SizeMB = [math]::Round($size/1MB, 1)
        LastWrite = $_.LastWriteTime.ToString('yyyy-MM-dd')
    }
} | Sort-Object SizeMB -Descending | Format-Table -AutoSize

Write-Host "`n=== logs/ subdirectory sizes ==="
Get-ChildItem -Path 'logs' -Directory | ForEach-Object {
    $size = (Get-ChildItem $_.FullName -Recurse -ErrorAction SilentlyContinue | Measure-Object -Property Length -Sum).Sum
    [PSCustomObject]@{
        Name = $_.Name
        SizeMB = [math]::Round($size/1MB, 1)
        LastWrite = $_.LastWriteTime.ToString('yyyy-MM-dd')
    }
} | Sort-Object SizeMB -Descending | Format-Table -AutoSize

Write-Host "`n=== docs/ subdirectory sizes ==="
Get-ChildItem -Path 'docs' -Directory | ForEach-Object {
    $size = (Get-ChildItem $_.FullName -Recurse -ErrorAction SilentlyContinue | Measure-Object -Property Length -Sum).Sum
    [PSCustomObject]@{
        Name = $_.Name
        SizeMB = [math]::Round($size/1MB, 1)
        LastWrite = $_.LastWriteTime.ToString('yyyy-MM-dd')
    }
} | Sort-Object SizeMB -Descending | Format-Table -AutoSize
