# Contrastive SWD ablation: mild / strong / extreme
$ErrorActionPreference = "Continue"
Set-Location "I:\Github\Latent_Style\SchrodingerBridge"
$env:PYTHONIOENCODING = "utf-8"

$cfgDir = "I:\Github\Latent_Style\SchrodingerBridge\configs"
$expDir = "I:\Github\Latent_Style\SchrodingerBridge\exp"
$testDir = "I:\datasets\wikiart_distinct5_samam_512_classview\test"
$cacheDir = "I:\Github\Latent_Style\SchrodingerBridge\exp\eval_cache"
$hfCache = "C:\Users\Administrator\.cache\huggingface\hub"
$dinoOut = "I:\Github\Latent_Style\SchrodingerBridge\exp\_dino_results"
$logOut = "C:\Users\Administrator\logs\cswd_ablation.out"
$epoch = "epoch_0005"

$names = @("cswd_mild", "cswd_strong", "cswd_extreme")

foreach ($name in $names) {
    $cfgFile = "$cfgDir\abl_$name.json"
    $saveDir = "$expDir\abl_$name"
    $ckpt = "$saveDir\$epoch.pt"
    $evalDir = "$saveDir\full_eval\$epoch"
    $imagesDir = "$evalDir\images"
    $dinoPath = "$dinoOut\abl_$name.json"

    Write-Output ""
    Write-Output "============================================================"
    Write-Output "=== ABLATION: $name START $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') ==="
    Write-Output "============================================================"

    if (-not (Test-Path $ckpt)) {
        Write-Output "  STEP 1: Training"
        python -u src\run.py --config $cfgFile 2>&1 | Tee-Object -FilePath $logOut -Append
        Write-Output "  STEP 1 DONE exit=$LASTEXITCODE $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')"
        if (-not (Test-Path $ckpt)) {
            Write-Output "  ERROR: Checkpoint not found: $ckpt"
            continue
        }
    } else {
        Write-Output "  STEP 1 SKIP: Checkpoint exists"
    }

    if ((Test-Path $ckpt) -and -not (Test-Path "$evalDir\summary.json")) {
        Write-Output "  STEP 2: Generate + CLIP-S/LPIPS eval"
        python -u src\utils\run_evaluation.py --checkpoint $ckpt --output $evalDir --test_dir $testDir --cache_dir $cacheDir --clip_hf_cache_dir $hfCache --eval_only_lpips_clip_style --eval_lpips_chunk_size 4 --batch_size 16 --metric_batch_size 16 --num_steps 8 2>&1 | Tee-Object -FilePath $logOut -Append
        Write-Output "  STEP 2 DONE exit=$LASTEXITCODE $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')"
    }

    if ((Test-Path "$evalDir\summary.json") -and -not (Test-Path $dinoPath)) {
        Write-Output "  STEP 3: DINO eval"
        python _compute_dino.py --images_dir $imagesDir --test_dir $testDir --dataset wikiart --output $dinoPath --max_refs 30 2>&1 | Tee-Object -FilePath $logOut -Append
        Write-Output "  STEP 3 DONE exit=$LASTEXITCODE $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')"
    }

    Write-Output "=== ABLATION: $name DONE $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') ==="
}

Write-Output ""
Write-Output "=== ALL CSWD ABLATIONS DONE $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') ==="
