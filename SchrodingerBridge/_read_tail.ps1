$lines = Get-Content "I:\Github\Latent_Style\SchrodingerBridge\remote_ablation_log.txt"
$total = $lines.Count
$start = [Math]::Max(0, $total - 40)
$lines[$start..($total-1)]
