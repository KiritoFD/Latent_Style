$ErrorActionPreference = "Continue"
Set-Location "I:\Github\Latent_Style\SchrodingerBridge"
$env:PYTHONPATH = "src"

# Find the images directory
$evalOut = "I:\Github\Latent_Style\SchrodingerBridge\exp\refactor_verify\t1_asg_5ep"
$imagesDir = $evalOut
if (Test-Path "$evalOut\images") {
    $imagesDir = "$evalOut\images"
}
Write-Output "Images dir: $imagesDir"
$imgCount = (Get-ChildItem $imagesDir -Filter *.png).Count
Write-Output "PNG count: $imgCount"

# Run DINO evaluation
Write-Output "=== Running DINO evaluation ==="
python _compute_dino.py `
    --images_dir $imagesDir `
    --test_dir "I:\datasets\wikiart_distinct5_samam_512_classview\test" `
    --dataset wikiart `
    --output "I:\Github\Latent_Style\SchrodingerBridge\state\dino\D5-512__t1_asg_5ep_retrain.json" `
    --hf_cache "I:\Github\Latent_Style\SchrodingerBridge\exp\eval_cache\hf" `
    --max_refs 30 `
    *>&1 | Tee-Object -FilePath "C:\Users\Administrator\logs\dino_eval_asg_retrain.log"

Write-Output "EXIT_CODE=$LASTEXITCODE"
