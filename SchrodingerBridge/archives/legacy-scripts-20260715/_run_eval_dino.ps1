# Eval + DINO only (training already done). Pass exp name as first arg.
$ErrorActionPreference = "Continue"
Set-Location "I:\Github\Latent_Style\SchrodingerBridge"
$env:PYTHONPATH = "."

$exp = $args[0]
if (-not $exp) { $exp = "d6_style_consist_15ep" }
$ckpt = "exp\$exp\epoch_0015.pt"
$evalDir = "exp\$exp\full_eval\epoch_0015"
$testDir = "I:\datasets\wikiart_distinct5_samam_512_classview\test"
$cacheDir = "I:\Github\Latent_Style\SchrodingerBridge\exp\eval_cache"
$hfCache = "C:\Users\Administrator\.cache\huggingface\hub"
$dinoOut = "exp\_dino_results\$exp.json"

Write-Output "=== EXP=$exp EVAL+DINO ==="
Write-Output "=== EVAL START $(Get-Date -Format 'HH:mm:ss') ==="
python -u src\utils\run_evaluation.py `
    --checkpoint $ckpt `
    --output $evalDir `
    --test_dir $testDir `
    --cache_dir $cacheDir `
    --clip_hf_cache_dir $hfCache `
    --batch_size 2 --generation_batch_size 2 --metric_batch_size 2 `
    --target_chunk_size 1 --vae_decode_batch_size 16 `
    --eval_only_lpips_clip_style --eval_lpips_chunk_size 4
$evalEc = $LASTEXITCODE
Write-Output "=== EVAL DONE exit=$evalEc $(Get-Date -Format 'HH:mm:ss') ==="

if ($evalEc -eq 0) {
    Write-Output "=== DINO START $(Get-Date -Format 'HH:mm:ss') ==="
    $imgDir = Join-Path $evalDir "images"
    python _compute_dino.py `
        --images_dir $imgDir `
        --test_dir $testDir `
        --dataset wikiart `
        --output $dinoOut `
        --hf_cache $hfCache `
        --max_refs 30
    Write-Output "=== DINO DONE exit=$LASTEXITCODE $(Get-Date -Format 'HH:mm:ss') ==="
}
Write-Output "=== ALL COMPLETE $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') ==="
