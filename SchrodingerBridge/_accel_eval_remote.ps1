$env:PYTHONPATH = "I:\Github\Latent_Style\SchrodingerBridge\src"
$env:CUDA_VISIBLE_DEVICES = "0"
Set-Location I:\Github\Latent_Style\SchrodingerBridge

# T11 4-step eval
Write-Host "=== T11 4-step eval ==="
python src\run.py --config configs\630_remote_t11_accel_4step.json 2>&1 | Tee-Object -FilePath "I:\Github\Latent_Style\SchrodingerBridge\exp\630_local_t11_long30ep\accel_4step.log"

# T11 2-step eval
Write-Host "=== T11 2-step eval ==="
python src\run.py --config configs\630_remote_t11_accel_2step.json 2>&1 | Tee-Object -FilePath "I:\Github\Latent_Style\SchrodingerBridge\exp\630_local_t11_long30ep\accel_2step.log"

Write-Host "=== All accel evals done ==="
