# Destructive ablation: disable one component at inference time, generate + evaluate on D5
$ErrorActionPreference = "Continue"
Set-Location "I:\Github\Latent_Style\SchrodingerBridge"
$env:PYTHONIOENCODING = "utf-8"

$ckpt = "I:\Github\Latent_Style\SchrodingerBridge\exp\t1_asg_5ep\epoch_0005.pt"
$testDir = "I:\datasets\wikiart_distinct5_samam_512_classview\test"
$cacheDir = "I:\Github\Latent_Style\SchrodingerBridge\exp\eval_cache"
$hfCache = "C:\Users\Administrator\.cache\huggingface\hub"
$cfgDir = "I:\Github\Latent_Style\SchrodingerBridge\configs"
$dinoOut = "I:\Github\Latent_Style\SchrodingerBridge\exp\_dino_results"
$logOut = "C:\Users\Administrator\logs\destructive_ablation.out"

# Ablation configs: name, config_override (or ""), num_steps_override (or "")
$ablations = @(
    @{ name = "wo_flow"; cfg = ""; steps = 1 },
    @{ name = "wo_asg"; cfg = "$cfgDir\ablation_wo_asg.json"; steps = 8 },
    @{ name = "wo_wavelet"; cfg = "$cfgDir\ablation_wo_wavelet.json"; steps = 8 },
    @{ name = "wo_spectral_ode"; cfg = "$cfgDir\ablation_wo_spectral_ode.json"; steps = 8 },
    @{ name = "wo_endpoint_adain"; cfg = "$cfgDir\ablation_wo_endpoint_adain.json"; steps = 8 }
)

foreach ($abl in $ablations) {
    $name = $abl.name
    $cfgFile = $abl.cfg
    $steps = $abl.steps
    $outDir = "I:\Github\Latent_Style\SchrodingerBridge\exp\ablation_destructive\$name\full_eval\epoch_0005"
    $imagesDir = "$outDir\images"
    $dinoPath = "$dinoOut\abl_$name.json"

    Write-Output ""
    Write-Output "=== ABLATION: $name START $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') ==="

    # Skip if already done (summary.json exists and DINO results exist)
    $skip = $false
    if ((Test-Path "$outDir\summary.json") -and (Test-Path $dinoPath)) {
        Write-Output "  SKIP: results already exist"
        $skip = $true
    }

    if (-not $skip) {
        # Step 1: Generate + CLIP-S/LPIPS eval
        Write-Output "  STEP 1: Generate + CLIP-S/LPIPS eval"
        $cmd = @(
            "python", "-u", "src\utils\run_evaluation.py",
            "--checkpoint", $ckpt,
            "--output", $outDir,
            "--test_dir", $testDir,
            "--cache_dir", $cacheDir,
            "--clip_hf_cache_dir", $hfCache,
            "--eval_only_lpips_clip_style",
            "--eval_lpips_chunk_size", "4",
            "--batch_size", "16",
            "--metric_batch_size", "16",
            "--num_steps", $steps
        )
        if ($cfgFile -ne "") {
            $cmd += "--config_override", $cfgFile
        }
        & $cmd[0] $cmd[1..($cmd.Length-1)] 2>&1 | Tee-Object -FilePath $logOut -Append
        Write-Output "  STEP 1 DONE exit=$LASTEXITCODE"
    }

    # Step 2: DINO eval (skip if exists)
    if (-not (Test-Path $dinoPath)) {
        Write-Output "  STEP 2: DINO eval"
        python _compute_dino.py `
            --images_dir $imagesDir `
            --test_dir $testDir `
            --dataset wikiart `
            --output $dinoPath `
            --max_refs 30 `
            2>&1 | Tee-Object -FilePath $logOut -Append
        Write-Output "  STEP 2 DONE exit=$LASTEXITCODE"
    } else {
        Write-Output "  STEP 2 SKIP: DINO results exist"
    }

    Write-Output "=== ABLATION: $name DONE $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') ==="
}

Write-Output ""
Write-Output "=== ALL ABLATIONS DONE $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') ==="
