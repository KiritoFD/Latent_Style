$ErrorActionPreference = 'SilentlyContinue'
Set-Location 'g:\GitHub\Latent_Style\SchrodingerBridge'

Write-Host "=== eval_cache/ contents ==="
Get-ChildItem -Path 'eval_cache' -Directory | ForEach-Object {
    $size = (Get-ChildItem $_.FullName -Recurse -ErrorAction SilentlyContinue | Measure-Object -Property Length -Sum).Sum
    [PSCustomObject]@{
        Name = $_.Name
        SizeMB = [math]::Round($size/1MB, 1)
        LastWrite = $_.LastWriteTime.ToString('yyyy-MM-dd')
    }
} | Sort-Object SizeMB -Descending | Format-Table -AutoSize

Write-Host "`n=== _codex_tmp/ contents ==="
Get-ChildItem -Path '_codex_tmp' -Recurse | Select-Object FullName, Length, LastWriteTime | Format-Table -AutoSize

Write-Host "`n=== archives/ contents ==="
Get-ChildItem -Path 'archives' -Recurse | Select-Object Name, @{N='SizeMB';E={[math]::Round($_.Length/1MB,2)}}, LastWriteTime | Format-Table -AutoSize

Write-Host "`n=== aaai2027/ contents (top level only) ==="
Get-ChildItem -Path 'aaai2027' | Select-Object Name, @{N='Type';E={if($_.PSIsContainer){'DIR'}else{'FILE'}}}, @{N='SizeMB';E={if($_.PSIsContainer){'-'}else{[math]::Round($_.Length/1MB,2)}}}, LastWriteTime | Format-Table -AutoSize

Write-Host "`n=== aaai_submission/ contents ==="
Get-ChildItem -Path 'aaai_submission' | Select-Object Name, @{N='Type';E={if($_.PSIsContainer){'DIR'}else{'FILE'}}}, @{N='SizeMB';E={if($_.PSIsContainer){'-'}else{[math]::Round($_.Length/1MB,2)}}}, LastWriteTime | Format-Table -AutoSize

Write-Host "`n=== scale/ top level ==="
Get-ChildItem -Path 'scale' | Select-Object Name, @{N='Type';E={if($_.PSIsContainer){'DIR'}else{'FILE'}}}, LastWriteTime | Format-Table -AutoSize

Write-Host "`n=== scale/datasets/ ==="
Get-ChildItem -Path 'scale\datasets' -Directory | ForEach-Object {
    $size = (Get-ChildItem $_.FullName -Recurse -ErrorAction SilentlyContinue | Measure-Object -Property Length -Sum).Sum
    [PSCustomObject]@{
        Name = $_.Name
        SizeMB = [math]::Round($size/1MB, 1)
        LastWrite = $_.LastWriteTime.ToString('yyyy-MM-dd')
    }
} | Sort-Object SizeMB -Descending | Format-Table -AutoSize
