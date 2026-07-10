Set-Location "I:\Github\Latent_Style\SchrodingerBridge"
$env:PYTHONIOENCODING = "utf-8"
python -u src\run.py --config configs\710_b0_weave_d5.json *> exp\710_b0_weave_log.txt
