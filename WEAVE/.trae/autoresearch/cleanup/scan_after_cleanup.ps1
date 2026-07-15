$ErrorActionPreference = 'SilentlyContinue'
Set-Location 'g:\GitHub\Latent_Style\SchrodingerBridge'

Write-Host "=== Remaining exp/ subdirs ==="
Get-ChildItem -Path 'exp' -Directory | ForEach-Object {
    $items = Get-ChildItem $_.FullName -Recurse -File -ErrorAction SilentlyContinue
    $size = 0
    if ($items) { $size = ($items | Measure-Object -Property Length -Sum).Sum }
    [PSCustomObject]@{
        Name = $_.Name
        SizeMB = [math]::Round($size/1MB, 1)
        LastWrite = $_.LastWriteTime.ToString('yyyy-MM-dd')
    }
} | Sort-Object SizeMB -Descending | Format-Table -AutoSize

Write-Host "`n=== Current top-level dirs ==="
Get-ChildItem -Directory | ForEach-Object {
    $items = Get-ChildItem $_.FullName -Recurse -File -ErrorAction SilentlyContinue
    $size = 0
    if ($items) { $size = ($items | Measure-Object -Property Length -Sum).Sum }
    [PSCustomObject]@{
        Name = $_.Name
        SizeMB = [math]::Round($size/1MB, 1)
        LastWrite = $_.LastWriteTime.ToString('yyyy-MM-dd')
    }
} | Sort-Object SizeMB -Descending | Format-Table -AutoSize
