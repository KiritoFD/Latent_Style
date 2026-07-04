$env:PYTHONPATH = "I:\Github\Latent_Style\SchrodingerBridge\src"
$env:CUDA_VISIBLE_DEVICES = "0"
Set-Location I:\Github\Latent_Style\SchrodingerBridge
New-Item -ItemType Directory -Force -Path "I:\Github\Latent_Style\SchrodingerBridge\exp\630_local_t11_long30ep" | Out-Null
Remove-Item -Path "I:\Github\Latent_Style\SchrodingerBridge\exp\630_local_t11_long30ep\train.log" -ErrorAction SilentlyContinue
python src\run.py --config configs\630_remote_t11_long30ep.json 2>&1 | Tee-Object -FilePath "I:\Github\Latent_Style\SchrodingerBridge\exp\630_local_t11_long30ep\train.log"
