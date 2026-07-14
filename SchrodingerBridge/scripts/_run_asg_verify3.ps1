$ErrorActionPreference = "Continue"
Set-Location "I:\Github\Latent_Style\SchrodingerBridge"
$env:PYTHONPATH = "src"

# Ensure output directory exists
$outDir = "I:\Github\Latent_Style\SchrodingerBridge\exp\refactor_verify\asg_activated"
if (-not (Test-Path $outDir)) {
    New-Item -ItemType Directory -Path $outDir -Force | Out-Null
}
if (-not (Test-Path "$outDir\images")) {
    New-Item -ItemType Directory -Path "$outDir\images" -Force | Out-Null
}

# Clear __pycache__
Get-ChildItem -Path "src" -Filter "__pycache__" -Directory -Recurse | Remove-Item -Recurse -Force -ErrorAction SilentlyContinue

# Re-run evaluation (will reuse existing 750 images since --force_regen is NOT set)
python run_evaluation.py `
    --checkpoint "I:\Github\Latent_Style\SchrodingerBridge\exp\t1_asg_5ep\epoch_0005.pt" `
    --output $outDir `
    --batch_size 2 `
    --ref_feature_batch_size 2 `
    --vae_decode_batch_size 16 `
    --test_dir "I:\datasets\wikiart_distinct5_samam_512_classview\test" `
    *>&1 | Tee-Object -FilePath "C:\Users\Administrator\logs\asg_verify3_eval.log"

Write-Output "EXIT_CODE=$LASTEXITCODE"
