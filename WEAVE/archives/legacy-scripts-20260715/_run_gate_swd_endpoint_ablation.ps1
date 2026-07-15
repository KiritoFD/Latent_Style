# Gate-Open + Sym-SWD + Endpoint-Only ablations
$ErrorActionPreference = "Continue"
Set-Location "I:\Github\Latent_Style\SchrodingerBridge"
$env:PYTHONIOENCODING = "utf-8"

$cfgDir = "I:\Github\Latent_Style\SchrodingerBridge\configs"
$expDir = "I:\Github\Latent_Style\SchrodingerBridge\exp"
$testDir = "I:\datasets\wikiart_distinct5_samam_512_classview\test"
$cacheDir = "I:\Github\Latent_Style\SchrodingerBridge\exp\eval_cache"
$hfCache = "C:\Users\Administrator\.cache\huggingface\hub"
$dinoOut = "I:\Github\Latent_Style\SchrodingerBridge\exp\_dino_results"
$logOut = "C:\Users\Administrator\logs\gate_swd_endpoint_ablation.out"
$baseCkpt = "I:\Github\Latent_Style\SchrodingerBridge\exp\t1_asg_5ep\epoch_0005.pt"

# === Experiment 1: Gate Open (train + eval) ===
$name = "gate_open"
$cfgFile = "$cfgDir\abl_$name.json"
$epoch = "epoch_0005"
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
    Write-Output "  STEP 1: Training (gate_init=3.0)"
    python -u src\run.py --config $cfgFile 2>&1 | Tee-Object -FilePath $logOut -Append
    Write-Output "  STEP 1 DONE exit=$LASTEXITCODE $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')"
    if (-not (Test-Path $ckpt)) {
        Write-Output "  ERROR: Checkpoint not found: $ckpt"
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

# === Experiment 2: Symmetric SWD (train + eval) ===
$name = "sym_swd"
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
    Write-Output "  STEP 1: Training (symmetric region SWD)"
    python -u src\run.py --config $cfgFile 2>&1 | Tee-Object -FilePath $logOut -Append
    Write-Output "  STEP 1 DONE exit=$LASTEXITCODE $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')"
    if (-not (Test-Path $ckpt)) {
        Write-Output "  ERROR: Checkpoint not found: $ckpt"
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

# === Experiment 3: Endpoint Only (inference-only, use base checkpoint + config override) ===
$name = "endpoint_only"
$overrideCfg = "$cfgDir\abl_$name.json"
$evalDir = "$expDir\abl_$name\full_eval\$epoch"
$imagesDir = "$evalDir\images"
$dinoPath = "$dinoOut\abl_$name.json"

Write-Output ""
Write-Output "============================================================"
Write-Output "=== ABLATION: $name START $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') ==="
Write-Output "============================================================"
Write-Output "  (inference-only, using base checkpoint: $baseCkpt)"

if (-not (Test-Path "$evalDir\summary.json")) {
    Write-Output "  STEP 2: Generate + CLIP-S/LPIPS eval (endpoint_adain_only_last_step=true)"
    python -u src\utils\run_evaluation.py --checkpoint $baseCkpt --config_override $overrideCfg --output $evalDir --test_dir $testDir --cache_dir $cacheDir --clip_hf_cache_dir $hfCache --eval_only_lpips_clip_style --eval_lpips_chunk_size 4 --batch_size 16 --metric_batch_size 16 --num_steps 8 2>&1 | Tee-Object -FilePath $logOut -Append
    Write-Output "  STEP 2 DONE exit=$LASTEXITCODE $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')"
} else {
    Write-Output "  STEP 2 SKIP: Summary exists"
}

if ((Test-Path "$evalDir\summary.json") -and -not (Test-Path $dinoPath)) {
    Write-Output "  STEP 3: DINO eval"
    python _compute_dino.py --images_dir $imagesDir --test_dir $testDir --dataset wikiart --output $dinoPath --max_refs 30 2>&1 | Tee-Object -FilePath $logOut -Append
    Write-Output "  STEP 3 DONE exit=$LASTEXITCODE $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')"
}

Write-Output "=== ABLATION: $name DONE $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') ==="

Write-Output ""
Write-Output "=== ALL ABLATIONS DONE $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') ==="
