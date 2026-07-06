# Kill stuck WD-VF eval python and aggregate result
$py = Get-Process -Id 4588 -ErrorAction SilentlyContinue
if ($py) {
    "Killing python 4588 (CPU=$($py.CPU)s WS=$([int]($py.WorkingSet64/1MB))MB)"
    Stop-Process -Id 4588 -Force
    Start-Sleep -Seconds 2
    $py2 = Get-Process -Id 4588 -ErrorAction SilentlyContinue
    if ($py2) { "  still running!" } else { "  killed successfully" }
} else {
    "python 4588 not found"
}

# Aggregate WD-VF
$csv = "I:\Github\Latent_Style\SchrodingerBridge\exp\wikiarts20_eval\metrics.csv"
if (Test-Path $csv) {
    $fi = Get-Item $csv
    "WD-VF csv size=$($fi.Length) mtime=$($fi.LastWriteTime)"
    $lines = Get-Content $csv
    "total lines: $($lines.Count)"
    "header: $($lines[0])"
    $clip_vals = @()
    $lpips_vals = @()
    $bad = 0
    for ($i = 1; $i -lt $lines.Count; $i++) {
        $cols = $lines[$i].Split(',')
        if ($cols.Count -ge 7) {
            $lp = 0.0
            $cl = 0.0
            if ([double]::TryParse($cols[4].Trim(), [ref]$lp) -and [double]::TryParse($cols[6].Trim(), [ref]$cl)) {
                $lpips_vals += $lp
                $clip_vals += $cl
            } else { $bad++ }
        }
    }
    if ($clip_vals.Count -gt 0) {
        $avg_clip = ($clip_vals | Measure-Object -Average).Average
        $avg_lpips = ($lpips_vals | Measure-Object -Average).Average
        "WD-VF 20-style SUMMARY: n=$($clip_vals.Count) CLIP-S=$avg_clip LPIPS=$avg_lpips (bad=$bad)"
    } else {
        "WD-VF no valid data (bad=$bad)"
    }
} else {
    "WD-VF csv NOT FOUND"
}

# Final summary of all 4 methods
"=== FINAL SUMMARY (all 4 methods, 20-style) ==="
$methods = @(
    @{name='identity'; csv="I:\Github\Latent_Style\SchrodingerBridge\exp\baseline_wikiarts20\identity\metrics.csv"},
    @{name='adain'; csv="I:\Github\Latent_Style\SchrodingerBridge\exp\baseline_wikiarts20\adain\metrics.csv"},
    @{name='wct'; csv="I:\Github\Latent_Style\SchrodingerBridge\exp\baseline_wikiarts20\wct\metrics.csv"},
    @{name='WD-VF'; csv="I:\Github\Latent_Style\SchrodingerBridge\exp\wikiarts20_eval\metrics.csv"}
)
foreach ($m in $methods) {
    if (Test-Path $m.csv) {
        $lines = Get-Content $m.csv
        $clip_vals = @()
        $lpips_vals = @()
        for ($i = 1; $i -lt $lines.Count; $i++) {
            $cols = $lines[$i].Split(',')
            if ($cols.Count -ge 7) {
                $lp = 0.0
                $cl = 0.0
                if ([double]::TryParse($cols[4].Trim(), [ref]$lp) -and [double]::TryParse($cols[6].Trim(), [ref]$cl)) {
                    $lpips_vals += $lp
                    $clip_vals += $cl
                }
            }
        }
        if ($clip_vals.Count -gt 0) {
            $avg_clip = ($clip_vals | Measure-Object -Average).Average
            $avg_lpips = ($lpips_vals | Measure-Object -Average).Average
            "$($m.name): CLIP-S=$avg_clip LPIPS=$avg_lpips n=$($clip_vals.Count)"
        }
    } else {
        "$($m.name): NO CSV"
    }
}
