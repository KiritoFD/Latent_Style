$ErrorActionPreference = "Continue"
Set-Location "I:\Github\Latent_Style\SchrodingerBridge"

# Baseline checkpoint (no ASG, clean code compatible)
$ckpt = "I:\Github\Latent_Style\SchrodingerBridge\exp\t1_asg_5ep_old_noasg\epoch_0005.pt"
$testDir = "I:\datasets\wikiart_distinct5_samam_512_classview\test"
$sweepRoot = "I:\Github\Latent_Style\SchrodingerBridge\exp\param_sweep"

# Clear __pycache__
Get-ChildItem -Path "src" -Filter "__pycache__" -Directory -Recurse | Remove-Item -Recurse -Force -ErrorAction SilentlyContinue

# Common eval args (VRAM safe: batch=2, ref_batch=2)
$commonArgs = @(
    '--checkpoint', $ckpt,
    '--batch_size', '2',
    '--ref_feature_batch_size', '2',
    '--vae_decode_batch_size', '16',
    '--test_dir', $testDir,
    '--force_regen'
)

# ============ Group A: Inference-time parameter sweep ============
# A1-A4: num_steps sweep (via --num_steps CLI arg)
$numStepsList = @(1, 4, 8, 12)
foreach ($ns in $numStepsList) {
    $expName = "a_steps_$ns"
    $outDir = "$sweepRoot\$expName"
    Write-Output "`n=== Running $expName (num_steps=$ns) ==="
    python run_evaluation.py @commonArgs --output $outDir --num_steps $ns *>&1 | Tee-Object -FilePath "C:\Users\Administrator\logs\sweep_$expName.log"
    Write-Output "EXIT_CODE_$expName=$LASTEXITCODE"
}

# A5-A8: style_extrap_alpha sweep (via config_override)
$extrapList = @(0.0, 0.1, 0.2, 0.5)
foreach ($alpha in $extrapList) {
    $expName = "a_extrap_$alpha"
    $outDir = "$sweepRoot\$expName"
    $overrideJson = "{""model"":{""style_extrap_alpha"":$alpha}}"
    $overrideFile = "$sweepRoot\_override_$expName.json"
    Set-Content -Path $overrideFile -Value $overrideJson -Encoding UTF8
    Write-Output "`n=== Running $expName (style_extrap_alpha=$alpha) ==="
    python run_evaluation.py @commonArgs --output $outDir --config_override $overrideFile *>&1 | Tee-Object -FilePath "C:\Users\Administrator\logs\sweep_$expName.log"
    Write-Output "EXIT_CODE_$expName=$LASTEXITCODE"
}

# A9-A11: endpoint_adain_scale sweep (via config_override)
$adainList = @(0.5, 1.0, 1.5)
foreach ($scale in $adainList) {
    $expName = "a_adain_$scale"
    $outDir = "$sweepRoot\$expName"
    $overrideJson = "{""model"":{""endpoint_adain_scale"":$scale}}"
    $overrideFile = "$sweepRoot\_override_$expName.json"
    Set-Content -Path $overrideFile -Value $overrideJson -Encoding UTF8
    Write-Output "`n=== Running $expName (endpoint_adain_scale=$scale) ==="
    python run_evaluation.py @commonArgs --output $outDir --config_override $overrideFile *>&1 | Tee-Object -FilePath "C:\Users\Administrator\logs\sweep_$expName.log"
    Write-Output "EXIT_CODE_$expName=$LASTEXITCODE"
}

Write-Output "`n=== ALL INFERENCE SWEEP DONE ==="
