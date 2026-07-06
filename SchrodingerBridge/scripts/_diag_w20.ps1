# Kill SaMam W20, check VRAM, diag eval failure
$ErrorActionPreference = "Continue"

Write-Host "=== Kill SaMam W20 process ==="
Get-CimInstance Win32_Process -Filter "Name='python.exe'" |
    Where-Object { $_.CommandLine -like "*_gen_samam_wiki20*" } |
    ForEach-Object {
        Write-Host "Killing PID $($_.ProcessId)"
        Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue
    }
Start-Sleep -Seconds 2

Write-Host ""
Write-Host "=== VRAM ==="
nvidia-smi --query-gpu=memory.used,memory.free,memory.total,utilization.gpu --format=csv

Write-Host ""
Write-Host "=== Run sdturbo_w20 eval manually with full output ==="
$PYTHON = "C:\Program Files\Python312\python.exe"
$USER_SITE = "C:\Users\Administrator\AppData\Roaming\Python\Python312\site-packages"
$REPO = "I:\Github\Latent_Style\SchrodingerBridge"
$env:HF_HOME = "C:\Users\Administrator\.cache\huggingface"
$env:TRANSFORMERS_OFFLINE = "1"
$env:TORCH_HOME = "C:\Users\Administrator\.cache\torch"
$env:PYTHONPATH = "$REPO\src;$USER_SITE;$REPO\scripts"
$env:PYTHONUSERBASE = "C:\Users\Administrator\AppData\Roaming\Python"
$env:CUDA_VISIBLE_DEVICES = "0"

& $PYTHON -u "$REPO\scripts\_eval_unified.py" `
    --image-dir "$REPO\exp\baseline_wikiarts20\sdturbo\images" `
    --dataset wiki20distinct5 `
    --output "$REPO\exp\_eval_sdturbo_w20.json" `
    --max-images 10 2>&1 |
    ForEach-Object { Write-Host $_ }
Write-Host "exit=$LASTEXITCODE"
