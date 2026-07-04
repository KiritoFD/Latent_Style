$env:PYTHONPATH = "I:\Github\Latent_Style\SchrodingerBridge\src"
$env:CUDA_VISIBLE_DEVICES = "0"
Set-Location I:\Github\Latent_Style\SchrodingerBridge
python src\run.py --config configs\630_remote_t11_long30ep_eval.json 2>&1 | Tee-Object -FilePath "I:\Github\Latent_Style\SchrodingerBridge\exp\630_local_t11_long30ep\eval_epoch30.log"
