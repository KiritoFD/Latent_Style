$env:PYTHONPATH = "I:\Github\Latent_Style\SchrodingerBridge\src"
$env:CUDA_VISIBLE_DEVICES = "0"
Set-Location I:\Github\Latent_Style\SchrodingerBridge

# T11 original 4-step eval
Write-Host "=== T11 orig 4-step eval ==="
python src\run.py --config configs\630_remote_t11_accel4_orig.json 2>&1 | Tee-Object -FilePath "I:\Github\Latent_Style\SchrodingerBridge\exp\accel_4step_orig.log"

# T11 original 2-step eval
Write-Host "=== T11 orig 2-step eval ==="
python src\run.py --config configs\630_remote_t11_accel2_orig.json 2>&1 | Tee-Object -FilePath "I:\Github\Latent_Style\SchrodingerBridge\exp\accel_2step_orig.log"

Write-Host "=== All orig accel evals done ==="
