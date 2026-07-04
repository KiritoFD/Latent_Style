$summaryPath = "I:\Github\Latent_Style\SchrodingerBridge\exp\clean_base_v2\full_eval\epoch_0010\summary.json"
$content = Get-Content $summaryPath -Raw
# Print first 3000 chars to see structure
Write-Output "=== summary.json (first 3000 chars) ==="
$content.Substring(0, [Math]::Min(3000, $content.Length))
Write-Output ""
Write-Output "=== Looking for key metrics ==="
# Search for clip_style, lpips in the content
$lines = $content -split "`n"
$lines | Where-Object { $_ -match "clip_style|lpips|clip_content|clip_dir|\"fid\"|pool|per_style|overall" } | Select-Object -First 30
