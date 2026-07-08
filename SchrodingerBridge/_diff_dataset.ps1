$lines = Get-Content "I:\Github\Latent_Style\SchrodingerBridge\src\utils\dataset.py"
$total = $lines.Count
$start = [Math]::Max(0, 360)
$end = [Math]::Min($total - 1, 420)
$lines[$start..$end] | ForEach-Object { "{0,4}: {1}" -f ($start + $_.ReadCount), $_ }
