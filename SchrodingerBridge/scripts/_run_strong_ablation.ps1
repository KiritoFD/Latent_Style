# Strong destructive ablation: retrain 4 configs + evaluate
$ErrorActionPreference = "Continue"
Set-Location "I:\Github\Latent_Style\SchrodingerBridge"
$env:PYTHONIOENCODING = "utf-8"

$cfgDir = "I:\Github\Latent_Style\SchrodingerBridge\configs"
$expDir = "I:\Github\Latent_Style\SchrodingerBridge\exp"
$testDir = "I:\datasets\wikiart_distinct5_samam_512_classview\test"
$cacheDir = "I:\Github\Latent_Style\SchrodingerBridge\exp\eval_cache"
$hfCache = "C:\Users\Administrator\.cache\huggingface\hub"
$dinoOut = "I:\Github\Latent_Style\SchrodingerBridge\exp\_dino_results"
$logOut = "C:\Users\Administrator\logs\strong_ablation.out"

# Ablation configs: name, config file, checkpoint epoch
$ablations = @(
    @{ name = "swd_to_mse";    cfg = "$cfgDir\abl_swd_to_mse.json";    epoch = "epoch_0005" },
    @{ name = "wo_wavelet";    cfg = "$cfgDir\abl_wo_wavelet.json";    epoch = "epoch_0005" },
    @{ name = "wo_swd";        cfg = "$cfgDir\abl_wo_swd.json";        epoch = "epoch_0005" },
    @{ name = "ll_equal";      cfg = "$cfgDir\abl_ll_equal.json";      epoch = "epoch_0005" }
)

foreach ($abl in $ablations) {
    $name = $abl.name
    $cfgFile = $abl.cfg
    $epoch = $abl.epoch
    $saveDir = "$expDir\abl_$name"
    $ckpt = "$saveDir\$epoch.pt"
    $evalDir = "$saveDir\full_eval\$epoch"
    $imagesDir = "$evalDir\images"
    $dinoPath = "$dinoOut\abl_$name.json"

    Write-Output ""
    Write-Output "============================================================"
    Write-Output "=== STRONG ABLATION: $name START $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') ==="
    Write-Output "============================================================"

    # Step 1: Train (skip if checkpoint exists)
    if (-not (Test-Path $ckpt)) {
        Write-Output "  STEP 1: Training"
        python -u src\run.py --config $cfgFile 2>&1 | Tee-Object -FilePath $logOut -Append
        Write-Output "  STEP 1 DONE exit=$LASTEXITCODE $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')"
        if (-not (Test-Path $ckpt)) {
            Write-Output "  ERROR: Checkpoint not found after training: $ckpt"
            Write-Output "  Skipping evaluation for $name"
            continue
        }
    } else {
        Write-Output "  STEP 1 SKIP: Checkpoint exists: $ckpt"
    }

    # Step 2: CLIP-S + LPIPS eval (skip if summary exists)
    $evalSkip = $false
    if (Test-Path "$evalDir\summary.json") {
        Write-Output "  STEP 2 SKIP: Summary exists"
        $evalSkip = $true
    }
    if (-not $evalSkip) {
        Write-Output "  STEP 2: Generate + CLIP-S/LPIPS eval"
        $cmd = @(
            "python", "-u", "src\utils\run_evaluation.py",
            "--checkpoint", $ckpt,
            "--output", $evalDir,
            "--test_dir", $testDir,
            "--cache_dir", $cacheDir,
            "--clip_hf_cache_dir", $hfCache,
            "--eval_only_lpips_clip_style",
            "--eval_lpips_chunk_size", "4",
            "--batch_size", "16",
            "--metric_batch_size", "16",
            "--num_steps", "8"
        )
        & $cmd[0] $cmd[1..($cmd.Length-1)] 2>&1 | Tee-Object -FilePath $logOut -Append
        Write-Output "  STEP 2 DONE exit=$LASTEXITCODE $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')"
    }

    # Step 3: DINO eval (skip if exists)
    if (-not (Test-Path $dinoPath)) {
        Write-Output "  STEP 3: DINO eval"
        python _compute_dino.py `
            --images_dir $imagesDir `
            --test_dir $testDir `
            --dataset wikiart `
            --output $dinoPath `
            --max_refs 30 `
            2>&1 | Tee-Object -FilePath $logOut -Append
        Write-Output "  STEP 3 DONE exit=$LASTEXITCODE $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')"
    } else {
        Write-Output "  STEP 3 SKIP: DINO results exist"
    }

    Write-Output "=== STRONG ABLATION: $name DONE $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') ==="
}

Write-Output ""
Write-Output "============================================================"
Write-Output "=== ALL STRONG ABLATIONS DONE $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') ==="
Write-Output "============================================================"
