# Foreground version: runs D10 directly so we can see errors
$env:P4_CKPT_PATH = "I:/Github/Latent_Style/SchrodingerBridge/exp/620_spectral_v2_weights/epoch_0001.pt"
$env:P4_CONFIG_PATH = "I:/Github/Latent_Style/SchrodingerBridge/configs/620_spectral_v2_weights.json"
$env:P4_BASELINE_CLIP = "0.6731"
$env:P4_BASELINE_LPIPS = "0.2781"
Set-Location I:/Github/Latent_Style/SchrodingerBridge
Write-Host "=== Running D10 in foreground ==="
Write-Host "CWD: $(Get-Location)"
Write-Host "P4_CKPT_PATH: $env:P4_CKPT_PATH"
& "C:/Program Files/Python312/python.exe" _p4_infer_ablation.py D10_b2_baseline avg_pool 0 0 single 0 0.3 0.3
Write-Host "=== Python exit code: $LASTEXITCODE ==="
