$logPath = "I:\Github\Latent_Style\SchrodingerBridge\exp\model_probe\target_hf_subband_ft6\logs\eval_adain20.log"
if (Test-Path $logPath) {
    Write-Output "=== LOG TAIL ==="
    Get-Content $logPath -Tail 15
} else {
    Write-Output "no log yet"
}
Write-Output "=== GPU ==="
nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader
Write-Output "=== PROCESS ==="
Get-Process python -ErrorAction SilentlyContinue | Select-Object Id, CPU, StartTime | Format-Table -AutoSize
