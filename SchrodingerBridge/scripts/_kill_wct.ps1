# Kill stuck wct eval python and aggregate result
$py = Get-Process -Id 2152 -ErrorAction SilentlyContinue
if ($py) {
    "Killing python 2152 (CPU=$($py.CPU)s WS=$([int]($py.WorkingSet64/1MB))MB)"
    Stop-Process -Id 2152 -Force
    Start-Sleep -Seconds 2
    $py2 = Get-Process -Id 2152 -ErrorAction SilentlyContinue
    if ($py2) { "  still running!" } else { "  killed successfully" }
} else {
    "python 2152 not found"
}

# Aggregate wct
$csv = "I:\Github\Latent_Style\SchrodingerBridge\exp\baseline_wikiarts20\wct\metrics.csv"
if (Test-Path $csv) {
    $fi = Get-Item $csv
    "wct csv size=$($fi.Length) mtime=$($fi.LastWriteTime)"
    $lines = Get-Content $csv
    "total lines: $($lines.Count)"
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
        "WCT 20-style SUMMARY: n=$($clip_vals.Count) CLIP-S=$avg_clip LPIPS=$avg_lpips (bad=$bad)"
    }
}
