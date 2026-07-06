# Launch SaMam W20 v2 in background via ps1 wrapper (schtasks-safe)
$ErrorActionPreference = "Continue"

$REPO = "I:\Github\Latent_Style\SchrodingerBridge"
$PYTHON = "C:\Program Files\Python312\python.exe"
$wrapper = "$REPO\scripts\_run_samam_w20_wrapper.ps1"

# Create wrapper script
@"
`$ErrorActionPreference = 'Continue'
`$env:HF_HOME = 'C:\Users\Administrator\.cache\huggingface'
`$env:TRANSFORMERS_OFFLINE = '1'
`$env:TORCH_HOME = 'C:\Users\Administrator\.cache\torch'
`$env:PYTHONPATH = '$REPO\src;C:\Users\Administrator\AppData\Roaming\Python\Python312\site-packages;$REPO\scripts'
`$env:PYTHONUSERBASE = 'C:\Users\Administrator\AppData\Roaming\Python'
`$env:CUDA_VISIBLE_DEVICES = '0'
& "$PYTHON" -u "$REPO\scripts\_gen_samam_wiki20_v2.py" *> "$REPO\logs\samam_w20_v2.log"
"@ | Out-File $wrapper -Encoding utf8

# Register schtasks
schtasks /Delete /TN "samam_w20_v2" /F 2>$null | Out-Null
$startDate = Get-Date -Format "yyyy/MM/dd"
schtasks /Create /TN "samam_w20_v2" `
    /TR "powershell.exe -NoProfile -ExecutionPolicy Bypass -File `"$wrapper`"" `
    /SC ONCE /ST 23:59 /SD $startDate `
    /RU SYSTEM /RL HIGHEST /F

schtasks /Run /TN "samam_w20_v2"

Start-Sleep -Seconds 5
Write-Host "=== Launched ==="
schtasks /Query /TN "samam_w20_v2" /FO LIST | Select-Object -First 5

Write-Host ""
Write-Host "=== VRAM ==="
nvidia-smi --query-gpu=memory.used,memory.free --format=csv
