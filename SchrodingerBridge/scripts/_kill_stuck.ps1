# Kill stuck python 4828 (identity eval already wrote csv, stuck in "Generating Summary")
$py = Get-Process -Id 4828 -ErrorAction SilentlyContinue
if ($py) {
    "Killing python 4828 (CPU=$($py.CPU)s WS=$([int]($py.WorkingSet64/1MB))MB)"
    Stop-Process -Id 4828 -Force
    Start-Sleep -Seconds 2
    $py2 = Get-Process -Id 4828 -ErrorAction SilentlyContinue
    if ($py2) { "  still running!" } else { "  killed successfully" }
} else {
    "python 4828 not found (already exited)"
}

# Verify identity csv is intact
$csv = "I:\Github\Latent_Style\SchrodingerBridge\exp\baseline_wikiarts20\identity\metrics.csv"
if (Test-Path $csv) {
    $fi = Get-Item $csv
    "identity csv size=$($fi.Length) mtime=$($fi.LastWriteTime)"
    # Count lines (quick, just first char check)
    $lineCount = 0
    $reader = [System.IO.File]::OpenText($csv)
    try {
        while ($reader.ReadLine() -ne $null) { $lineCount++ }
    } finally { $reader.Close() }
    "identity csv lines: $lineCount (expected 12001 = 1 header + 12000 data)"
}

# Check adain/wct ready for eval
foreach ($m in @('adain','wct')) {
    $imgDir = "I:\Github\Latent_Style\SchrodingerBridge\exp\baseline_wikiarts20\$m\images"
    $done = "I:\Github\Latent_Style\SchrodingerBridge\exp\baseline_wikiarts20\$m\_DONE"
    $imgCount = (Get-ChildItem $imgDir -Filter "*.png" -ErrorAction SilentlyContinue).Count
    $doneExists = Test-Path $done
    "$m => images=$imgCount _DONE=$doneExists"
}
