# Phase T2: Frequency-aware ASG — train + full_eval + DINO
$ErrorActionPreference = "Continue"
Set-Location "I:\Github\Latent_Style\SchrodingerBridge"
$env:PYTHONIOENCODING = "utf-8"

$exp = "t2_fasg_5ep"
$config = "configs\$exp.json"
$ckpt = "exp\$exp\epoch_0005.pt"
$evalDir = "exp\$exp\full_eval\epoch_0005"
$testDir = "I:\datasets\wikiart_distinct5_samam_512_classview\test"
$cacheDir = "I:\Github\Latent_Style\SchrodingerBridge\exp\eval_cache"
$hfCache = "C:\Users\Administrator\.cache\huggingface\hub"
$dinoOut = "exp\_dino_results\$exp.json"
$logOut = "C:\Users\Administrator\logs\${exp}_train_eval.out"

Write-Output "=== EXP=$exp ==="
Write-Output "=== TRAIN START $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') ==="
python -u src\run.py --config $config 2>&1 | Tee-Object -FilePath $logOut
$trainEc = $LASTEXITCODE
Write-Output "=== TRAIN DONE exit=$trainEc $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') ==="
if ($trainEc -ne 0 -or -not (Test-Path $ckpt)) {
    Write-Output "FATAL: training failed or no checkpoint. Aborting."
    exit 1
}

Write-Output "=== EVAL START $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') ==="
python -u src\utils\run_evaluation.py `
    --checkpoint $ckpt `
    --output $evalDir `
    --test_dir $testDir `
    --cache_dir $cacheDir `
    --clip_hf_cache_dir $hfCache `
    --batch_size 2 --generation_batch_size 2 --metric_batch_size 2 `
    --target_chunk_size 1 --vae_decode_batch_size 16 `
    --eval_only_lpips_clip_style --eval_lpips_chunk_size 4 2>&1 | Tee-Object -FilePath $logOut -Append
$evalEc = $LASTEXITCODE
Write-Output "=== EVAL DONE exit=$evalEc $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') ==="

if ($evalEc -eq 0) {
    Write-Output "=== DINO START $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') ==="
    $imgDir = Join-Path $evalDir "images"
    python _compute_dino.py `
        --images_dir $imgDir `
        --test_dir $testDir `
        --dataset wikiart `
        --output $dinoOut `
        --hf_cache $hfCache `
        --max_refs 30 2>&1 | Tee-Object -FilePath $logOut -Append
    Write-Output "=== DINO DONE exit=$LASTEXITCODE $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') ==="
}
Write-Output "=== ALL COMPLETE $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') ==="
