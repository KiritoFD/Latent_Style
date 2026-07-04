Set-Location C:\Users\Administrator
if (-not (Test-Path logs)) { New-Item -ItemType Directory -Path logs -Force | Out-Null }
python run.py --config configs\630_latent_256.json *> logs\latent256_train.log
