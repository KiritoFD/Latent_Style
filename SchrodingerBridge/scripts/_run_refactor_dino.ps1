$ErrorActionPreference = "Continue"
Set-Location "I:\Github\Latent_Style\SchrodingerBridge"
$env:PYTHONPATH = "src"
python -m utils.compute_dino_metrics `
    --eval_dir "I:\Github\Latent_Style\SchrodingerBridge\exp\refactor_verify\t1_asg_5ep" `
    --test_dir "I:\datasets\wikiart_distinct5_samam_512_classview\test" `
    --cache_dir "I:\Github\Latent_Style\eval_cache\hf" `
    --batch_size 4 `
    --max_refs_per_style 30 `
    --exclude_source_from_style_refs `
    *>&1 | Tee-Object -FilePath "C:\Users\Administrator\logs\refactor_verify_dino.log"
Write-Output "EXIT_CODE=$LASTEXITCODE"
