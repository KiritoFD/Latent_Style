Set-Location "I:\Github\Latent_Style\SchrodingerBridge"
$env:PYTHONPATH = "."
python _compute_dino.py `
    --images_dir "exp\abl_no_ll_fm\full_eval\epoch_0015\images" `
    --test_dir "I:\datasets\wikiart_distinct5_samam_512_classview\test" `
    --dataset wikiart `
    --output "exp\_dino_results\abl_no_ll_fm.json" `
    --hf_cache "C:\Users\Administrator\.cache\huggingface\hub" `
    --max_refs 30
