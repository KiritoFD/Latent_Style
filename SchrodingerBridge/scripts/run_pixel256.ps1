Set-Location C:\Users\Administrator
if (-not (Test-Path logs)) { New-Item -ItemType Directory -Path logs -Force | Out-Null }
python run.py --config configs\630_pixel_256.json *> logs\pixel256_train.log
