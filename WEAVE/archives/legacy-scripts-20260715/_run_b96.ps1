Set-Location "I:\Github\Latent_Style\SchrodingerBridge"
$env:PYTHONIOENCODING = "utf-8"
python -u src\run.py --config configs\infra_train_b96.json 2>&1 | Tee-Object -FilePath "C:\Users\Administrator\logs\infra_train_b96.out"
