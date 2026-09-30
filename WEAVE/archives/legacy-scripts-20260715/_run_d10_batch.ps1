# D10 batch runner: sequential train+eval+DINO for d10a, d10b, d10c
# Usage: powershell -File _run_d10_batch.ps1
$ErrorActionPreference = "Continue"
Set-Location "I:\Github\Latent_Style\SchrodingerBridge"
$env:PYTHONPATH = "."

$testDir = "I:\datasets\wikiart_distinct5_samam_512_classview\test"
$cacheDir = "I:\Github\Latent_Style\SchrodingerBridge\exp\eval_cache"
$hfCache = "C:\Users\Administrator\.cache\huggingface\hub"

$exps = @("d10a_dim48_15ep", "d10b_dim48_gate05_15ep", "d10c_dim32_15ep")

foreach ($exp in $exps) {
    $config = "configs\$exp.json"
    $ckpt = "exp\$exp\epoch_0015.pt"
    $evalDir = "exp\$exp\full_eval\epoch_0015"
    $dinoOut = "exp\_dino_results\$exp.json"
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

    # EVAL
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

    # DINO
    if ($evalEc -eq 0) {
        Write-Output "=== DINO START $(Get-Date -Format 'HH:mm:ss') ==="
        $imgDir = Join-Path $evalDir "images"
        python _compute_dino.py `
            --images_dir $imgDir `
            --test_dir $testDir `
            --dataset wikiart `
            --output $dinoOut `
            --hf_cache $hfCache `
            --max_refs 30 2>&1 | Tee-Object -FilePath $logOut -Append
        Write-Output "=== DINO DONE exit=$LASTEXITCODE $(Get-Date -Format 'HH:mm:ss') ==="
    }
    Write-Output "=== EXP=$exp COMPLETE $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') ==="
}
Write-Output "=== ALL D10 BATCH COMPLETE $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') ==="
