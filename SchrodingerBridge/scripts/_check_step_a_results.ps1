$evalDir = "I:\Github\Latent_Style\SchrodingerBridge\exp\model_probe\target_hf_subband_ft6\full_eval\adain20"
Write-Output "=== FILES IN EVAL DIR ==="
Get-ChildItem $evalDir -ErrorAction SilentlyContinue | Select-Object Name, Length, LastWriteTime | Format-Table -AutoSize
Write-Output "=== SUMMARY JSON ==="
$summaryPath = "$evalDir\clip_lpips_summary.json"
if (Test-Path $summaryPath) {
    Get-Content $summaryPath
} else {
    Write-Output "no clip_lpips_summary.json"
}
Write-Output ""
Write-Output "=== DINO JSON ==="
$dinoPath = "$evalDir\dino.json"
if (Test-Path $dinoPath) {
    Get-Content $dinoPath
} else {
    Write-Output "no dino.json"
}
Write-Output ""
Write-Output "=== GPU ==="
nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader
Write-Output "=== PYTHON PROCS ==="
Get-Process python -ErrorAction SilentlyContinue | Select-Object Id, CPU, StartTime | Format-Table -AutoSize
