Set-Location "I:\Github\Latent_Style\SchrodingerBridge"
$env:PYTHONIOENCODING = "utf-8"
python -u src\run.py --config configs\exp_sty_stage7_delta.json 2>&1 | Tee-Object -FilePath "C:\Users\Administrator\logs\sty_inject\stage7_delta_train.out"
