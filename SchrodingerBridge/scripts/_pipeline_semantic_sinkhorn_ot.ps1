$ErrorActionPreference = "Continue"
$Root = "I:\Github\Latent_Style\SchrodingerBridge"
$Py = "C:\Program Files\Python312\python.exe"
Set-Location $Root
$env:PYTHONIOENCODING = "utf-8"

$TestDir = "I:\datasets\wikiart_distinct5_samam_512_classview\test"
$HfCache = "C:\Users\Administrator\.cache\huggingface\hub"

$Tag = "dino_s_break\semantic_sinkhorn_ot"
$Config = "exp_semantic_sinkhorn_ot.json"
$ConfigOverride = "$Root\configs\eval_adain_20.json"

$CkptDir = "$Root\exp\$Tag"
$LogDir = "$CkptDir\logs"
New-Item -ItemType Directory -Force -Path $LogDir | Out-Null

# ===== TRAIN =====
Write-Output "=== [$Tag] TRAIN START $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') ==="
& $Py -u "$Root\src\run.py" --config "$Root\configs\$Config" 2>&1 | Tee-Object -FilePath "$LogDir\train.log"
if ($LASTEXITCODE -ne 0) { Write-Output "=== [$Tag] TRAIN FAILED (exit=$LASTEXITCODE) ==="; exit 1 }
Write-Output "=== [$Tag] TRAIN DONE $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') ==="

# ===== EVAL CLIP/LPIPS (adain=2.0) =====
$Ckpt = "$CkptDir\epoch_0010.pt"
if (-not (Test-Path $Ckpt)) {
    $Ckpt = Get-ChildItem "$CkptDir\epoch_*.pt" | Sort-Object Name -Descending | Select-Object -First 1 -ExpandProperty FullName
    Write-Output "=== [$Tag] using ckpt: $Ckpt ==="
}
Write-Output "=== [$Tag] EVAL START $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') ==="
& $Py -u "$Root\src\utils\run_evaluation.py" `
    --config_override $ConfigOverride `
    --checkpoint $Ckpt --output $CkptDir `
    --save_generated_images --batch_size 2 `
    --ref_feature_batch_size 2 --clip_hf_cache_dir $HfCache 2>&1 | Tee-Object -FilePath "$LogDir\eval.log"
if ($LASTEXITCODE -ne 0) { Write-Output "=== [$Tag] EVAL FAILED (exit=$LASTEXITCODE) ==="; exit 1 }
Write-Output "=== [$Tag] EVAL DONE $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') ==="

# ===== DINO =====
Write-Output "=== [$Tag] DINO START $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') ==="
& $Py -u "$Root\_compute_dino.py" `
    --images_dir "$CkptDir\images" --test_dir $TestDir --dataset wikiart `
    --output "$CkptDir\dino.json" --hf_cache $HfCache --max_refs 30 2>&1 | Tee-Object -FilePath "$LogDir\dino.log"
if ($LASTEXITCODE -ne 0) { Write-Output "=== [$Tag] DINO FAILED (exit=$LASTEXITCODE) ==="; exit 1 }

Write-Output "=== [$Tag] ALL COMPLETE $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') ==="
Write-Output "===== RESULTS ====="
if (Test-Path "$CkptDir\summary.json") {
    $sum = Get-Content "$CkptDir\summary.json" -Raw | ConvertFrom-Json
    $ov = $sum.analysis.all_pairs_overview
    Write-Output "CLIP-S = $($ov.clip_style)"
    Write-Output "LPIPS  = $($ov.content_lpips)"
}
if (Test-Path "$CkptDir\dino.json") {
    $dino = Get-Content "$CkptDir\dino.json" -Raw | ConvertFrom-Json
    Write-Output "DINO-C = $($dino.dino_content)"
    Write-Output "DINO-S = $($dino.dino_style)"
}
Write-Output "===== END ====="
