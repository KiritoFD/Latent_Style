# Reproduce main-table metrics + fp32 AMP-off ablation
# bf16: reproduce DINO-S=0.4859, CLIP-S=0.7075, LPIPS=0.2583, DINO-C=0.8287
# fp32: test if higher precision breaks DINO-S ceiling
$ErrorActionPreference = "Continue"
$Root = "I:\Github\Latent_Style\SchrodingerBridge"
$Py = "C:\Program Files\Python312\python.exe"
Set-Location $Root
$env:PYTHONIOENCODING = "utf-8"

$TestDir = "I:\datasets\wikiart_distinct5_samam_512_classview\test"
$HfCache = "C:\Users\Administrator\.cache\huggingface\hub"

function Run-One([string]$Tag, [string]$Config) {
    $CkptDir = "$Root\exp\repro\$Tag"
    $LogDir = "$CkptDir\logs"
    New-Item -ItemType Directory -Force -Path $LogDir | Out-Null

    Write-Output "=== [$Tag] TRAIN START $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') ==="
    & $Py -u "$Root\src\run.py" --config "$Root\configs\$Config" 2>&1 > "$LogDir\train.log"
    if ($LASTEXITCODE -ne 0) {
        Write-Output "=== [$Tag] TRAIN FAILED exit=$LASTEXITCODE ==="
        return
    }
    Write-Output "=== [$Tag] TRAIN DONE $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') ==="

    $Ckpt = "$CkptDir\epoch_0010.pt"
    if (-not (Test-Path $Ckpt)) {
        Write-Output "=== [$Tag] CHECKPOINT MISSING: $Ckpt ==="
        return
    }

    Write-Output "=== [$Tag] EVAL START (AdaIN=2.0) $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') ==="
    & $Py -u "$Root\src\utils\run_evaluation.py" `
        --config_override "$Root\configs\eval_adain_20.json" `
        --checkpoint $Ckpt `
        --output $CkptDir `
        --save_generated_images `
        --batch_size 2 `
        --ref_feature_batch_size 2 `
        --clip_hf_cache_dir $HfCache 2>&1 > "$LogDir\eval.log"
    if ($LASTEXITCODE -ne 0) {
        Write-Output "=== [$Tag] EVAL FAILED exit=$LASTEXITCODE ==="
        return
    }
    Write-Output "=== [$Tag] EVAL DONE $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') ==="

    Write-Output "=== [$Tag] DINO START $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') ==="
    & $Py -u "$Root\_compute_dino.py" `
        --images_dir "$CkptDir\images" --test_dir $TestDir --dataset wikiart `
        --output "$CkptDir\dino.json" --hf_cache $HfCache --max_refs 30 2>&1 > "$LogDir\dino.log"
    if ($LASTEXITCODE -ne 0) {
        Write-Output "=== [$Tag] DINO FAILED exit=$LASTEXITCODE ==="
        return
    }
    Write-Output "=== [$Tag] DINO DONE $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') ==="
    Write-Output "=== [$Tag] ALL COMPLETE $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') ==="
}

Run-One "bf16" "repro_bf16.json"
Run-One "fp32" "repro_fp32.json"

Write-Output "=== REPRO PIPELINE COMPLETE $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') ==="
