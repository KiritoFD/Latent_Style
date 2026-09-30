$ErrorActionPreference = "Continue"
Set-Location "I:\Github\Latent_Style\SchrodingerBridge"
$env:PYTHONPATH = "src"

# Clear __pycache__
Get-ChildItem -Path "src" -Filter "__pycache__" -Directory -Recurse | Remove-Item -Recurse -Force -ErrorAction SilentlyContinue

# Run evaluation with T1 ASG checkpoint to verify ASG restoration
python run_evaluation.py `
    --checkpoint "I:\Github\Latent_Style\SchrodingerBridge\exp\t1_asg_5ep\epoch_0005.pt" `
    --output "I:\Github\Latent_Style\SchrodingerBridge\exp\asg_restore_verify\t1_asg_5ep" `
    --batch_size 2 `
    --ref_feature_batch_size 2 `
    --vae_decode_batch_size 16 `
    --test_dir "I:\datasets\wikiart_distinct5_samam_512_classview\test" `
    *>&1 | Tee-Object -FilePath "C:\Users\Administrator\logs\asg_restore_eval.log"

Write-Output "EXIT_CODE=$LASTEXITCODE"
