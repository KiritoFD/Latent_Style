# 3-seed experiment: train 3 seeds on D5-512, eval each on D5/P2A/R5
# Runs sequentially on remote RTX 3060. Logs to C:\Users\Administrator\logs\seed3\
$ErrorActionPreference = "Continue"
Set-Location "I:\Github\Latent_Style\SchrodingerBridge"
$env:PYTHONIOENCODING = "utf-8"

$logDir = "C:\Users\Administrator\logs\seed3"
if (-not (Test-Path $logDir)) { New-Item -ItemType Directory -Path $logDir -Force | Out-Null }

$cacheDir = "I:\Github\Latent_Style\SchrodingerBridge\exp\eval_cache"
$hfCache = "C:\Users\Administrator\.cache\huggingface\hub"

# Dataset configs
$datasets = @(
    @{name="d5";   testDir="I:\datasets\wikiart_distinct5_samam_512_classview\test"; styles="Early_Renaissance,Impressionism,Minimalism,Rococo,Ukiyo_e"; dataset="wikiart"},
    @{name="p2a";  testDir="I:\datasets\legacy256_overfit50\test";                    styles="cezanne,Hayao,monet,photo,vangogh";                       dataset="p2a"},
    @{name="r5";   testDir="I:\datasets\wikiarts20_512_test";                         styles="Cubism,Expressionism,Pop_Art,Romanticism,Symbolism";      dataset="wikiart"}
)

$seeds = @(
    @{name="seed42";   config="configs\seed3\seed42_b96.json";   ckpt="exp\seed3\seed42_b96\epoch_0005.pt"},
    @{name="seed123";  config="configs\seed3\seed123_b96.json";  ckpt="exp\seed3\seed123_b96\epoch_0005.pt"},
    @{name="seed2024"; config="configs\seed3\seed2024_b96.json"; ckpt="exp\seed3\seed2024_b96\epoch_0005.pt"}
)

# ===== PHASE 1: TRAINING =====
foreach ($seed in $seeds) {
    $trainLog = "$logDir\$($seed.name)_train.out"
    Write-Output ""
    Write-Output "=== TRAIN $($seed.name) START $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') ==="
    & python -u src\run.py --config $seed.config 2>&1 | Tee-Object -FilePath $trainLog
    Write-Output "=== TRAIN $($seed.name) DONE exit=$LASTEXITCODE $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') ==="

    if (-not (Test-Path $seed.ckpt)) {
        Write-Output "ERROR: checkpoint $($seed.ckpt) not found after training. Skipping evals."
        continue
    }
}

# ===== PHASE 2: EVALUATION (3 seeds x 3 datasets = 9 evals) =====
foreach ($seed in $seeds) {
    if (-not (Test-Path $seed.ckpt)) {
        Write-Output "SKIP $($seed.name): checkpoint missing."
        continue
    }

    foreach ($ds in $datasets) {
        $evalDir = "exp\seed3\$($seed.name)_$($ds.name)_eval"
        $evalLog = "$logDir\$($seed.name)_$($ds.name)_eval.out"

        Write-Output ""
        Write-Output "=== EVAL $($seed.name) on $($ds.name) START $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') ==="

        # Step 1: inference + CLIP/LPIPS
        $summaryPath = "$evalDir\full_eval\epoch_0005\summary.json"
        if (-not (Test-Path $summaryPath)) {
            $evalArgs = @(
                "-u", "src\utils\run_evaluation.py",
                "--checkpoint", $seed.ckpt,
                "--output", "$evalDir\full_eval\epoch_0005",
                "--test_dir", $ds.testDir,
                "--style_subdirs", $ds.styles,
                "--cache_dir", $cacheDir,
                "--clip_hf_cache_dir", $hfCache,
                "--batch_size", "2", "--generation_batch_size", "2", "--metric_batch_size", "2",
                "--target_chunk_size", "1", "--vae_decode_batch_size", "16",
                "--eval_only_lpips_clip_style", "--eval_lpips_chunk_size", "4"
            )
            & python @evalArgs 2>&1 | Tee-Object -FilePath $evalLog
        } else {
            Write-Output "  summary.json exists, skipping inference."
        }

        # Step 2: DINO eval
        $dinoOut = "exp\seed3\_dino\$($seed.name)_$($ds.name).json"
        if (-not (Test-Path $dinoOut)) {
            $dinoArgs = @(
                "_compute_dino.py",
                "--images_dir", "$evalDir\full_eval\epoch_0005\images",
                "--test_dir", $ds.testDir,
                "--dataset", $ds.dataset,
                "--output", $dinoOut,
                "--hf_cache", $hfCache,
                "--max_refs", "30"
            )
            if ($ds.name -eq "r5") {
                $dinoArgs += @("--style_subdirs", $ds.styles)
            }
            & python @dinoArgs 2>&1 | Tee-Object -FilePath $evalLog -Append
        } else {
            Write-Output "  DINO result exists, skipping."
        }

        Write-Output "=== EVAL $($seed.name) on $($ds.name) DONE exit=$LASTEXITCODE $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') ==="
    }
}

Write-Output ""
Write-Output "=== ALL 3-SEED EXPERIMENTS COMPLETE $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') ==="
