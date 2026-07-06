# Quick identity CSV check
$csv = "I:\Github\Latent_Style\SchrodingerBridge\exp\baseline_wikiarts20\identity\metrics.csv"
if (Test-Path $csv) {
    $fi = Get-Item $csv
    "identity csv size=$($fi.Length) mtime=$($fi.LastWriteTime)"
    $lines = Get-Content $csv
    "total lines: $($lines.Count)"
    "header: $($lines[0])"
    # Parse CLIP-S (col 6) and LPIPS (col 4)
    $clip_vals = @()
    $lpips_vals = @()
    for ($i = 1; $i -lt $lines.Count; $i++) {
        $cols = $lines[$i].Split(',')
        if ($cols.Count -ge 7) {
            $lpips_vals += [double]$cols[3]
            $clip_vals += [double]$cols[5]
        }
    }
    if ($clip_vals.Count -gt 0) {
        $avg_clip = ($clip_vals | Measure-Object -Average).Average
        $avg_lpips = ($lpips_vals | Measure-Object -Average).Average
        "identity 20-style: n=$($clip_vals.Count) CLIP-S=$avg_clip LPIPS=$avg_lpips"
    }
} else {
    "identity csv NOT FOUND"
}

# Check adain/wct image counts
foreach ($m in @('adain','wct')) {
    $dir = "I:\Github\Latent_Style\SchrodingerBridge\exp\baseline_wikiarts20\$m\images"
    if (Test-Path $dir) {
        $count = (Get-ChildItem $dir -Filter "*.png").Count
        "$m images: $count"
        $done = "I:\Github\Latent_Style\SchrodingerBridge\exp\baseline_wikiarts20\$m\_DONE"
        if (Test-Path $done) { "$m _DONE exists" } else { "$m _DONE MISSING" }
    } else {
        "$m images dir NOT FOUND"
    }
}

# Check if python 4828 still running
$py = Get-Process -Id 4828 -ErrorAction SilentlyContinue
if ($py) {
    "python 4828 still running: CPU=$($py.CPU) WS=$([int]($py.WorkingSet64/1MB))MB"
} else {
    "python 4828 NOT running"
}
