Set-Location 'I:\Github\Latent_Style\SchrodingerBridge'
$task = 'LANCET_transport_texton_hayao'
try {
    Get-ScheduledTask -TaskName $task | Select-Object TaskName, State | Format-List | Out-String -Width 200
} catch {
    "task missing: $($_.Exception.Message)"
}
nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits
$root = 'exp\vae_backend\ema_transport_texton_hayao'
if (Test-Path "$root\task.log") {
    '--- task.log ---'
    Get-Content "$root\task.log" -Tail 80
}
if (Test-Path "$root\vae_backend_256_results.csv") {
    '--- csv ---'
    Get-Content "$root\vae_backend_256_results.csv"
}
if (Test-Path $root) {
    '--- latest cross_by_target_style ---'
    Get-ChildItem $root -Filter summary.json -Recurse -ErrorAction SilentlyContinue |
        Sort-Object LastWriteTime -Descending |
        Select-Object -First 3 |
        ForEach-Object {
            "summary: $($_.FullName)"
            try {
                $j = Get-Content -Raw -LiteralPath $_.FullName | ConvertFrom-Json
                $cross = $j.analysis.cross_by_target_style
                if ($cross) {
                    $cross.PSObject.Properties | ForEach-Object {
                        $name = $_.Name
                        $v = $_.Value
                        '{0} cross style={1:N5} lpips={2:N5} images={3}' -f $name, [double]$v.clip_style, [double]$v.content_lpips, [int]$v.image_count
                    }
                } else {
                    $metrics = Join-Path $_.DirectoryName 'metrics.csv'
                    if (-not (Test-Path $metrics)) {
                        $metrics = Join-Path $_.DirectoryName 'metrics_reuse_generated.csv'
                    }
                    if (Test-Path $metrics) {
                        "cross_by_target_style missing; fallback from $metrics"
                        $rows = Import-Csv $metrics
                        $targets = $rows | Select-Object -ExpandProperty tgt_style -Unique | Sort-Object
                        foreach ($target in $targets) {
                            $rs = @($rows | Where-Object { $_.tgt_style -eq $target -and $_.src_style -ne $_.tgt_style })
                            if ($rs.Count -gt 0) {
                                $style = ($rs | Measure-Object -Property clip_style -Average).Average
                                $lpips = ($rs | Measure-Object -Property content_lpips -Average).Average
                                '{0} cross style={1:N5} lpips={2:N5} images={3}' -f $target, [double]$style, [double]$lpips, [int]$rs.Count
                            }
                        }
                    } else {
                        'cross_by_target_style missing'
                    }
                }
            } catch {
                "summary parse failed: $($_.Exception.Message)"
            }
        }
}
if (Test-Path $root) {
    '--- dirs ---'
    Get-ChildItem $root -Directory -ErrorAction SilentlyContinue | Select-Object Name, LastWriteTime | Format-Table -AutoSize | Out-String -Width 200
}
