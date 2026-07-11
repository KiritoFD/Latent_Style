$ErrorActionPreference = "Continue"
Set-Location "I:\Github\Latent_Style\SchrodingerBridge"

# Clear __pycache__
Get-ChildItem -Path "src" -Filter "__pycache__" -Directory -Recurse | Remove-Item -Recurse -Force -ErrorAction SilentlyContinue

$ckpt = "I:\Github\Latent_Style\SchrodingerBridge\exp\refactor_clean_baseline\epoch_0005.pt"
$testDir = "I:\datasets\wikiart_distinct5_samam_512_classview\test"
$outDir = "I:\Github\Latent_Style\SchrodingerBridge\exp\refactor_clean_baseline\eval"

# Phase 1: CLIP-S + LPIPS evaluation
Write-Output "=== Phase 1: CLIP-S + LPIPS evaluation ==="
python run_evaluation.py `
    --checkpoint $ckpt `
    --output $outDir `
    --batch_size 2 `
    --ref_feature_batch_size 2 `
    --vae_decode_batch_size 16 `
    --test_dir $testDir `
    --force_regen `
    *>&1 | Tee-Object -FilePath "C:\Users\Administrator\logs\eval_clean_baseline.log"
Write-Output "EVAL_EXIT_CODE=$LASTEXITCODE"

# Phase 2: DINO evaluation
Write-Output "`n=== Phase 2: DINO evaluation ==="
$imagesDir = $outDir
if (Test-Path "$outDir\images") {
    $imagesDir = "$outDir\images"
}
python _compute_dino.py `
    --images_dir $imagesDir `
    --test_dir $testDir `
    --dataset wikiart `
    --output "I:\Github\Latent_Style\SchrodingerBridge\state\dino\D5-512__refactor_clean_baseline.json" `
    --hf_cache "I:\Github\Latent_Style\SchrodingerBridge\exp\eval_cache\hf" `
    --max_refs 30 `
    *>&1 | Tee-Object -FilePath "C:\Users\Administrator\logs\dino_clean_baseline.log"
Write-Output "DINO_EXIT_CODE=$LASTEXITCODE"
