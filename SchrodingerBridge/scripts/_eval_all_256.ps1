# Re-evaluate all 256 baselines using unified eval (consistent LPIPS)
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

$targets = @(
    @{ name = "adain_256";   dir = "I:\exp_256_photo2art\adain_256\images" },
    @{ name = "wct_256";     dir = "I:\exp_256_photo2art\wct_256\images" },
    @{ name = "samst_256";   dir = "I:\exp_256_photo2art\samst_256\images" },
    @{ name = "samam_256";   dir = "I:\exp_256_photo2art\samam_256\images" },
    @{ name = "identity_256";dir = "I:\exp_256_photo2art\identity_256\images" }
)

foreach ($t in $targets) {
    if (-not (Test-Path $t.dir)) {
        Write-Host "[$($t.name)] SKIP: dir not found"
        continue
    }
    $cnt = (Get-ChildItem $t.dir -Filter *.png).Count + (Get-ChildItem $t.dir -Filter *.jpg).Count
    if ($cnt -eq 0) {
        Write-Host "[$($t.name)] SKIP: 0 images"
        continue
    }
    Write-Host "[$($t.name)] Evaluating $cnt images..."
    $evalOut = "$REPO\exp\_eval_$($t.name)_unified.json"
    & $PYTHON -u "$REPO\scripts\_eval_unified.py" `
        --image-dir $t.dir `
        --dataset photo2art256 `
        --output $evalOut `
        --max-images 750
    if (Test-Path $evalOut) {
        $r = Get-Content $evalOut -Raw | ConvertFrom-Json
        Write-Host "  CLIP-S=$($r.clip_s)  LPIPS=$($r.lpips)  MUSIQ=$($r.musiq)"
    }
}
