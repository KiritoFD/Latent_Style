$summaryPath = "I:\Github\Latent_Style\SchrodingerBridge\exp\clean_base_v2\full_eval\epoch_0010\summary.json"
$content = Get-Content $summaryPath -Raw
Write-Output "=== summary.json length: $($content.Length) ==="
Write-Output ""
Write-Output "=== First 2000 chars ==="
$content.Substring(0, [Math]::Min(2000, $content.Length))
Write-Output ""
Write-Output "=== Searching for metrics ==="
$lines = $content -split "`n"
$lineNum = 0
foreach ($line in $lines) {
    $lineNum++
    if ($line -match "clip_style" -or $line -match "lpips" -or $line -match "clip_content" -or $line -match "clip_dir" -or $line -match "pool" -or $line -match "overall" -or $line -match "per_style") {
        Write-Output "${lineNum}: $($line.Trim())"
    }
}
