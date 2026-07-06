# W20 full generation: SDTurbo first (~6h), then SaMam v3 fp16 (~50h)
# VRAM-optimized: batch_size=1, fp16, empty_cache every image
$ErrorActionPreference = "Continue"

$REPO = "I:\Github\Latent_Style\SchrodingerBridge"
$PYTHON = "C:\Program Files\Python312\python.exe"
$USER_SITE = "C:\Users\Administrator\AppData\Roaming\Python\Python312\site-packages"
$env:HF_HOME = "C:\Users\Administrator\.cache\huggingface"
$env:TRANSFORMERS_OFFLINE = "1"
$env:TORCH_HOME = "C:\Users\Administrator\.cache\torch"
$env:PYTHONPATH = "$REPO\src;$USER_SITE;$REPO\scripts"
$env:PYTHONUSERBASE = "C:\Users\Administrator\AppData\Roaming\Python"
$env:CUDA_VISIBLE_DEVICES = "0"

$masterLog = "$REPO\logs\w20_full_gen.log"
"=== W20 FULL GEN START $(Get-Date -Format 'yyyy-MM-ddTHH:mm:ss') ===" | Out-File $masterLog -Encoding utf8

# Phase 1: SD-Turbo W20 full (20 styles, ~6h)
Write-Host "=== Phase 1: SD-Turbo W20 full ==="
"[$(Get-Date -Format 'HH:mm:ss')] Phase 1: SD-Turbo W20 full" | Out-File $masterLog -Append -Encoding utf8
& $PYTHON -u "$REPO\scripts\_gen_sdturbo_w20_full.py" 2>&1 |
    Out-File "$REPO\logs\sdturbo_w20_full.log" -Encoding utf8
"[$(Get-Date -Format 'HH:mm:ss')] Phase 1 DONE" | Out-File $masterLog -Append -Encoding utf8

# Phase 2: SaMam W20 v3 fp16 (20 styles, ~50h)
Write-Host "=== Phase 2: SaMam W20 v3 ==="
"[$(Get-Date -Format 'HH:mm:ss')] Phase 2: SaMam W20 v3" | Out-File $masterLog -Append -Encoding utf8
& $PYTHON -u "$REPO\scripts\_gen_samam_w20_v3.py" 2>&1 |
    Out-File "$REPO\logs\samam_w20_v3.log" -Encoding utf8
"[$(Get-Date -Format 'HH:mm:ss')] Phase 2 DONE" | Out-File $masterLog -Append -Encoding utf8

"=== W20 FULL GEN END $(Get-Date -Format 'yyyy-MM-ddTHH:mm:ss') ===" | Out-File $masterLog -Append -Encoding utf8
Write-Host "=== ALL DONE ==="
