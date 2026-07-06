# Eval W20 MUSIQ only (reuse existing CLIP-S/LPIPS from previous eval)
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

$logFile = "$REPO\logs\eval_w20_musiq.log"
"=== W20 MUSIQ START $(Get-Date -Format 'yyyy-MM-ddTHH:mm:ss') ===" | Out-File $logFile -Encoding utf8

# Only MUSIQ needed (CLIP-S/LPIPS already computed); use --skip-clip --skip-lpips
$targets = @(
    @{ name = "sdturbo_w20"; dir = "$REPO\exp\baseline_wikiarts20\sdturbo\images" },
    @{ name = "styleid_w20"; dir = "$REPO\exp\baseline_wikiarts20\styleid\images" },
    @{ name = "samst_w20";   dir = "$REPO\exp\baseline_wikiarts20\samst\images" },
    @{ name = "cut_w20";     dir = "$REPO\exp\baseline_v2\images\cut" }
)

foreach ($t in $targets) {
    if (-not (Test-Path $t.dir)) {
        "[$($t.name)] SKIP: dir not found" | Out-File $logFile -Append -Encoding utf8
        continue
    }
    $out = "$REPO\exp\_eval_$($t.name)_musiq.json"
    "[$($t.name)] START $(Get-Date -Format 'HH:mm:ss')" | Out-File $logFile -Append -Encoding utf8
    & $PYTHON -u "$REPO\scripts\_eval_unified.py" `
        --image-dir $t.dir `
        --dataset wiki20distinct5 `
        --output $out `
        --max-images 750 `
        --skip-clip --skip-lpips 2>&1 | Out-File $logFile -Append -Encoding utf8
    if (Test-Path $out) {
        $r = Get-Content $out -Raw | ConvertFrom-Json
        $msg = "  MUSIQ=$($r.musiq)"
        Write-Host $msg
        $msg | Out-File $logFile -Append -Encoding utf8
    }
    "[$($t.name)] DONE" | Out-File $logFile -Append -Encoding utf8
}

"=== W20 MUSIQ END $(Get-Date -Format 'yyyy-MM-ddTHH:mm:ss') ===" | Out-File $logFile -Append -Encoding utf8
Write-Host "=== DONE ==="
