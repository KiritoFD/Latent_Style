# Evaluate existing CUT 512 images (baseline_v2/images/cut) using unified eval
# Naming: {src_style}__{src_stem}__to__{tgt_style}.png  (matches _eval_unified.py)
$ErrorActionPreference = "Continue"

$REPO = "I:\Github\Latent_Style\SchrodingerBridge"
$PYTHON = "C:\Program Files\Python312\python.exe"
$USER_SITE = "C:\Users\Administrator\AppData\Roaming\Python\Python312\site-packages"
$env:HF_HOME = "C:\Users\Administrator\.cache\huggingface"
$env:TRANSFORMERS_OFFLINE = "1"
$env:TORCH_HOME = "C:\Users\Administrator\.cache\torch"
$env:PYTHONPATH = "$REPO\src;$USER_SITE;$REPO\scripts"
$env:PYTHONUSERBASE = "C:\Users\Administrator\AppData\Roaming\Python"

# CUT 512 distinct5 images already exist
$cut512Dir = "$REPO\exp\baseline_v2\images\cut"
$evalOut = "$REPO\exp\_eval_cut_w20.json"

Write-Host "=== Evaluating CUT WikiArt-20 (distinct5) ==="
Write-Host "  image_dir: $cut512Dir"
$cnt = (Get-ChildItem $cut512Dir -Filter *.png).Count + (Get-ChildItem $cut512Dir -Filter *.jpg).Count
Write-Host "  image count: $cnt"

& $PYTHON -u "$REPO\scripts\_eval_unified.py" `
    --image-dir $cut512Dir `
    --dataset wiki20distinct5 `
    --output $evalOut `
    --max-images 750

Write-Host "=== Done. Results: $evalOut ==="
if (Test-Path $evalOut) { Get-Content $evalOut -Raw }
