# Phase T4: Adaptive WCT scales — eval on T1 ASG checkpoint + DINO
# Usage: powershell -File _run_t4_adaptive_wct.ps1
$ErrorActionPreference = "Continue"
Set-Location "I:\Github\Latent_Style\SchrodingerBridge"
$env:PYTHONIOENCODING = "utf-8"

$exp = "t4_asg_adaptive_wct"
$ckpt = "exp\t1_asg_5ep\epoch_0005.pt"
$evalDir = "exp\$exp\full_eval\epoch_0005"
$overrideCfg = "configs\t4_override.json"
$testDir = "I:\datasets\wikiart_distinct5_samam_512_classview\test"
$cacheDir = "I:\Github\Latent_Style\SchrodingerBridge\exp\eval_cache"
$hfCache = "C:\Users\Administrator\.cache\huggingface\hub"
$dinoOut = "exp\_dino_results\$exp.json"
$logOut = "C:\Users\Administrator\logs\${exp}_eval.out"

Write-Output "=== EXP=$exp ==="
Write-Output "=== EVAL START $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') ==="
python -u src\utils\run_evaluation.py `
    --checkpoint $ckpt `
    --output $evalDir `
    --config_override $overrideCfg `
    --test_dir $testDir `
    --cache_dir $cacheDir `
    --clip_hf_cache_dir $hfCache `
    --batch_size 2 --generation_batch_size 2 --metric_batch_size 2 `
    --target_chunk_size 1 --vae_decode_batch_size 16 `
    --eval_only_lpips_clip_style --eval_lpips_chunk_size 4 2>&1 | Tee-Object -FilePath $logOut
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
