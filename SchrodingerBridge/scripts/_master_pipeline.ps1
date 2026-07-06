# Master: run eval_all_unified.ps1, then SaMam W20 v2, then eval samam_w20
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

$masterLog = "$REPO\logs\master_pipeline.log"
"=== MASTER START $(Get-Date -Format 'yyyy-MM-ddTHH:mm:ss') ===" | Out-File $masterLog -Encoding utf8

# Phase 1: unified evals (W20 + 256)
Write-Host "=== Phase 1: eval_all_unified ==="
"[$(Get-Date -Format 'HH:mm:ss')] Phase 1: eval_all_unified" | Out-File $masterLog -Append -Encoding utf8
& powershell -NoProfile -ExecutionPolicy Bypass -File "$REPO\scripts\_eval_all_unified.ps1" 2>&1 |
    Out-File $masterLog -Append -Encoding utf8
"[$(Get-Date -Format 'HH:mm:ss')] Phase 1 DONE" | Out-File $masterLog -Append -Encoding utf8

# Phase 2: SaMam W20 generation (v2 fixed)
Write-Host "=== Phase 2: SaMam W20 v2 ==="
"[$(Get-Date -Format 'HH:mm:ss')] Phase 2: SaMam W20 v2" | Out-File $masterLog -Append -Encoding utf8
& $PYTHON -u "$REPO\scripts\_gen_samam_wiki20_v2.py" 2>&1 |
    Out-File "$REPO\logs\samam_w20_v2.log" -Encoding utf8
"[$(Get-Date -Format 'HH:mm:ss')] Phase 2 DONE" | Out-File $masterLog -Append -Encoding utf8

# Phase 3: eval SaMam W20
Write-Host "=== Phase 3: eval SaMam W20 ==="
"[$(Get-Date -Format 'HH:mm:ss')] Phase 3: eval samam_w20" | Out-File $masterLog -Append -Encoding utf8
$samamW20Dir = "$REPO\exp\baseline_wikiarts20\samam\images"
if ((Test-Path $samamW20Dir) -and ((Get-ChildItem $samamW20Dir -File).Count -ge 100)) {
    & $PYTHON -u "$REPO\scripts\_eval_unified.py" `
        --image-dir $samamW20Dir `
        --dataset wiki20distinct5 `
        --output "$REPO\exp\_eval_samam_w20.json" `
        --max-images 750 --skip-musiq 2>&1 |
        Out-File $masterLog -Append -Encoding utf8
} else {
    "samam_w20: SKIP (insufficient images after gen)" | Out-File $masterLog -Append -Encoding utf8
}
"[$(Get-Date -Format 'HH:mm:ss')] Phase 3 DONE" | Out-File $masterLog -Append -Encoding utf8

"=== MASTER END $(Get-Date -Format 'yyyy-MM-ddTHH:mm:ss') ===" | Out-File $masterLog -Append -Encoding utf8
Write-Host "=== MASTER DONE ==="

# Disable the watchdog if we created one
$wd = "abl512_watchdog"
schtasks /End /TN $wd 2>$null
schtasks /Change /TN $wd /Disable 2>$null
