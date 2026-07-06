# Launch SaMam Random5 fp16 via schtasks
$ErrorActionPreference = "Continue"

$REPO = "I:\Github\Latent_Style\SchrodingerBridge"
$PYTHON = "C:\Program Files\Python312\python.exe"

# Create wrapper
$wrapper = "$REPO\scripts\_run_samam_random5_wrapper.ps1"
@"
`$ErrorActionPreference = 'Continue'
`$env:HF_HOME = 'C:\Users\Administrator\.cache\huggingface'
`$env:TRANSFORMERS_OFFLINE = '1'
`$env:TORCH_HOME = 'C:\Users\Administrator\.cache\torch'
`$env:PYTHONPATH = '$REPO\src;C:\Users\Administrator\AppData\Roaming\Python\Python312\site-packages;$REPO\scripts'
`$env:PYTHONUSERBASE = 'C:\Users\Administrator\AppData\Roaming\Python'
`$env:CUDA_VISIBLE_DEVICES = '0'
& "$PYTHON" -u "$REPO\scripts\_gen_samam_random5_fp16.py" *> "$REPO\logs\samam_random5_fp16.log"
"@ | Out-File $wrapper -Encoding utf8

# Register schtasks
schtasks /Delete /TN "samam_random5" /F 2>$null | Out-Null
$startDate = Get-Date -Format "yyyy/MM/dd"
schtasks /Create /TN "samam_random5" `
    /TR "powershell.exe -NoProfile -ExecutionPolicy Bypass -File `"$wrapper`"" `
    /SC ONCE /ST 23:59 /SD $startDate `
    /RU SYSTEM /RL HIGHEST /F

schtasks /Run /TN "samam_random5"

Start-Sleep -Seconds 10
Write-Host "=== Launched ==="
schtasks /Query /TN "samam_random5" /FO LIST | Select-Object -First 5

Write-Host ""
Write-Host "=== VRAM ==="
nvidia-smi --query-gpu=memory.used,memory.free --format=csv

Write-Host ""
Write-Host "=== Log tail ==="
$log = "$REPO\logs\samam_random5_fp16.log"
if (Test-Path $log) { Get-Content $log -Tail 10 }
