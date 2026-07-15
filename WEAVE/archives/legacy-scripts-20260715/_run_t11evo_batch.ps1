# T11 evolution batch runner: sequential train+eval for t11_repro, t11e1, t11e2, t11e3
# Usage: powershell -File _run_t11evo_batch.ps1
$ErrorActionPreference = "Continue"
Set-Location "I:\Github\Latent_Style\SchrodingerBridge"
$env:PYTHONPATH = "."

$testDir = "I:\datasets\wikiart_distinct5_samam_512_classview\test"
$cacheDir = "I:\Github\Latent_Style\SchrodingerBridge\exp\eval_cache"
$hfCache = "C:\Users\Administrator\.cache\huggingface\hub"

$exps = @("t11_repro_15ep", "t11e1_ll05_15ep", "t11e2_extrap05_15ep", "t11e3_p09_15ep")

foreach ($exp in $exps) {
    $config = "configs\$exp.json"
    # Find final epoch checkpoint (epoch_0015 for 15ep configs)
    $ckpt = "exp\$exp\epoch_0015.pt"
    $evalDir = "exp\$exp\full_eval\epoch_0015"
    $logOut = "C:\Users\Administrator\logs\${exp}_train_eval.out"

    Write-Output "=== EXP=$exp START $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') ==="

    # TRAIN
    Write-Output "=== TRAIN START $(Get-Date -Format 'HH:mm:ss') ==="
    python -u src\run.py --config $config 2>&1 | Tee-Object -FilePath $logOut
    $trainEc = $LASTEXITCODE
    Write-Output "=== TRAIN DONE exit=$trainEc $(Get-Date -Format 'HH:mm:ss') ==="
    if ($trainEc -ne 0 -or -not (Test-Path $ckpt)) {
        Write-Output "FATAL: training failed or no checkpoint for $exp. Skipping to next."
        continue
    }

    # EVAL (only clip-style + lpips, no DINO for now)
    Write-Output "=== EVAL START $(Get-Date -Format 'HH:mm:ss') ==="
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
    Write-Output "=== EVAL DONE exit=$evalEc $(Get-Date -Format 'HH:mm:ss') ==="

    Write-Output "=== EXP=$exp COMPLETE $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') ==="
}
Write-Output "=== ALL T11 EVO BATCH COMPLETE $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') ==="
