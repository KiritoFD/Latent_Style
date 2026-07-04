$dir = "I:\Github\Latent_Style\SchrodingerBridge\exp\p4_fusion_breakout\infer_ablation"
Write-Host "=== All inference ablation result files ==="
Get-ChildItem -Path $dir -Filter *.json | Sort-Object Name | ForEach-Object {
    Write-Host $_.Name
}

Write-Host ""
Write-Host "=== T4 results (sorted by clip_style) ==="
$t4Files = Get-ChildItem -Path $dir -Filter "T4*.json"
$results = @()
foreach ($f in $t4Files) {
    try {
        $data = Get-Content $f.FullName -Raw | ConvertFrom-Json
        $metrics = $data.metrics
        $results += [PSCustomObject]@{
            Name = $data.exp_name
            ClipStyle = $metrics.allpairs_clip_style
            Lpips = $metrics.allpairs_content_lpips
            TransferClip = $metrics.transfer_clip_style
            TransferLpips = $metrics.transfer_content_lpips
            Params = ($data.params | ConvertTo-Json -Compress)
        }
    } catch {
        Write-Host "Failed to parse: $($f.Name) - $_"
    }
}
$results | Sort-Object ClipStyle -Descending | Format-Table -AutoSize

Write-Host ""
Write-Host "=== All results sorted by clip_style (top 15) ==="
$allFiles = Get-ChildItem -Path $dir -Filter "*.json" | Where-Object { $_.Name -notmatch "_override" }
$allResults = @()
foreach ($f in $allFiles) {
    try {
        $data = Get-Content $f.FullName -Raw | ConvertFrom-Json
        $metrics = $data.metrics
        if ($metrics.allpairs_clip_style -ne $null) {
            $allResults += [PSCustomObject]@{
                Name = $data.exp_name
                ClipStyle = $metrics.allpairs_clip_style
                Lpips = $metrics.allpairs_content_lpips
            }
        }
    } catch {}
}
$allResults | Sort-Object ClipStyle -Descending | Select-Object -First 15 | Format-Table -AutoSize
