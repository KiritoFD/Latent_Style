# Run SaMam Wiki20 generation + evaluation using patched SS2D_Encoder (fallback to torch)
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

# Verify patch is applied
Write-Host "=== Applying clean SS2D patch ==="
& $PYTHON "I:\Github\Latent_Style\SchrodingerBridge\scripts\_patch_samam_v2.py"

# Phase 1: Generate SaMam W20 images
Write-Host "`n=== Phase 1: SaMam Wiki20 generation ==="
$samamW20Dir = "$REPO\exp\baseline_wikiarts20\samam\images"
New-Item -ItemType Directory -Force -Path $samamW20Dir | Out-Null
$cnt = (Get-ChildItem $samamW20Dir -Filter *.png -ErrorAction SilentlyContinue).Count
Write-Host "  existing: $cnt/750"
if ($cnt -lt 750) {
    & $PYTHON -u "$REPO\scripts\_gen_samam_wiki20.py" 2>&1 | Tee-Object -FilePath "$REPO\logs\samam_w20.gen.log"
    $cnt = (Get-ChildItem $samamW20Dir -Filter *.png -ErrorAction SilentlyContinue).Count
    Write-Host "  final: $cnt/750"
}

# Phase 2: Evaluate
Write-Host "`n=== Phase 2: Evaluate SaMam W20 ==="
$evalOut = "$REPO\exp\_eval_samam_w20.json"
& $PYTHON -u "$REPO\scripts\_eval_unified.py" `
    --image-dir $samamW20Dir `
    --dataset wiki20distinct5 `
    --output $evalOut `
    --max-images 750

Write-Host "`n=== Done ==="
if (Test-Path $evalOut) {
    Write-Host "Results:"
    Get-Content $evalOut -Raw
}
