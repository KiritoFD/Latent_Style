# Inference-only parameter sweep using baseline checkpoint.
# Tests different endpoint_adain_scale and style_extrap_alpha values.
$ErrorActionPreference = "Continue"
Set-Location "I:\Github\Latent_Style\SchrodingerBridge"
$env:PYTHONPATH = "."

$ckpt = "exp\hp_simple_swd12_15ep\epoch_0015.pt"
$testDir = "I:\datasets\wikiart_distinct5_samam_512_classview\test"
$cacheDir = "I:\Github\Latent_Style\SchrodingerBridge\exp\eval_cache"
$hfCache = "C:\Users\Administrator\.cache\huggingface\hub"

# Override configs to test
$overrides = @(
    @{name="wct_strong"; file="scripts\_overrides\wct_strong.json"},
    @{name="wct_ll"; file="scripts\_overrides\wct_ll.json"},
    @{name="extrap06"; file="scripts\_overrides\extrap06.json"},
    @{name="combo06"; file="scripts\_overrides\combo06.json"}
)

foreach ($ov in $overrides) {
    $name = $ov.name
    $ovFile = $ov.file
    $evalDir = "exp\hp_simple_swd12_15ep\full_eval\sweep_$name"
    $dinoOut = "exp\_dino_results\sweep_$name.json"
    $logOut = "C:\Users\Administrator\logs\sweep_$name.out"

    Write-Output "=== SWEEP=$name START $(Get-Date -Format 'HH:mm:ss') ==="

    # Evaluation with override
    python -u src\utils\run_evaluation.py `
        --checkpoint $ckpt `
        --output $evalDir `
        --test_dir $testDir `
        --cache_dir $cacheDir `
        --clip_hf_cache_dir $hfCache `
        --config_override $ovFile `
        --batch_size 2 --generation_batch_size 2 --metric_batch_size 2 `
        --target_chunk_size 1 --vae_decode_batch_size 16 `
        --eval_only_lpips_clip_style --eval_lpips_chunk_size 4 2>&1 | Tee-Object -FilePath $logOut

    $evalEc = $LASTEXITCODE
    Write-Output "=== EVAL DONE exit=$evalEc $(Get-Date -Format 'HH:mm:ss') ==="

    if ($evalEc -eq 0) {
        # DINO computation
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
    Write-Output "=== SWEEP=$name COMPLETE ==="
}
Write-Output "=== ALL SWEEPS COMPLETE $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') ==="
