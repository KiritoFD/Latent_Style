# Run all unified evals in sequence: W20 + 256
# W20: skip MUSIQ (table doesn't have it); 256: full eval
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

$logFile = "$REPO\logs\eval_all_unified.log"
"=== eval_all_unified START $(Get-Date -Format 'yyyy-MM-ddTHH:mm:ss') ===" | Out-File $logFile -Encoding utf8

function Run-Eval($name, $imgDir, $dataset, $outFile, $skipMusiq) {
    $cmd = "& `"$PYTHON`" -u `"$REPO\scripts\_eval_unified.py`" --image-dir `"$imgDir`" --dataset $dataset --output `"$outFile`" --max-images 750"
    if ($skipMusiq) { $cmd += " --skip-musiq" }
    Write-Host "[$name] $cmd"
    "[$name] START $(Get-Date -Format 'HH:mm:ss')" | Out-File $logFile -Append -Encoding utf8
    Invoke-Expression $cmd 2>&1 | Out-File $logFile -Append -Encoding utf8
    if (Test-Path $outFile) {
        $r = Get-Content $outFile -Raw | ConvertFrom-Json
        $msg = "  CLIP-S=$($r.clip_s)  LPIPS=$($r.lpips)  MUSIQ=$($r.musiq)"
        Write-Host $msg
        $msg | Out-File $logFile -Append -Encoding utf8
        "[$name] DONE" | Out-File $logFile -Append -Encoding utf8
    } else {
        Write-Host "[$name] FAILED - no output"
        "[$name] FAILED" | Out-File $logFile -Append -Encoding utf8
    }
}

# === W20 evals (skip MUSIQ, table doesn't have MUSIQ column) ===
Run-Eval "sdturbo_w20" "$REPO\exp\baseline_wikiarts20\sdturbo\images" "wiki20distinct5" "$REPO\exp\_eval_sdturbo_w20.json" $true
Run-Eval "styleid_w20"  "$REPO\exp\baseline_wikiarts20\styleid\images"  "wiki20distinct5" "$REPO\exp\_eval_styleid_w20.json"  $true
Run-Eval "samst_w20"    "$REPO\exp\baseline_wikiarts20\samst\images"    "wiki20distinct5" "$REPO\exp\_eval_samst_w20.json"    $true
# SaMam W20 only if images exist
$samamW20Dir = "$REPO\exp\baseline_wikiarts20\samam\images"
if ((Test-Path $samamW20Dir) -and ((Get-ChildItem $samamW20Dir -File).Count -ge 100)) {
    Run-Eval "samam_w20" $samamW20Dir "wiki20distinct5" "$REPO\exp\_eval_samam_w20.json" $true
} else {
    "samam_w20: SKIP (not enough images)" | Out-File $logFile -Append -Encoding utf8
    Write-Host "samam_w20: SKIP (not enough images)"
}

# === 256 evals (full, with MUSIQ) ===
$base256 = "I:\exp_256_photo2art"
Run-Eval "adain_256"    "$base256\adain_256\images"    "photo2art256" "$REPO\exp\_eval_adain_256_unified.json"    $false
Run-Eval "wct_256"      "$base256\wct_256\images"      "photo2art256" "$REPO\exp\_eval_wct_256_unified.json"      $false
Run-Eval "samst_256"    "$base256\samst_256\images"    "photo2art256" "$REPO\exp\_eval_samst_256_unified.json"    $false
Run-Eval "samam_256"    "$base256\samam_256\images"    "photo2art256" "$REPO\exp\_eval_samam_256_unified.json"    $false
Run-Eval "identity_256" "$base256\identity_256\images" "photo2art256" "$REPO\exp\_eval_identity_256_unified.json" $false

"=== eval_all_unified END $(Get-Date -Format 'yyyy-MM-ddTHH:mm:ss') ===" | Out-File $logFile -Append -Encoding utf8
Write-Host "=== ALL DONE ==="
