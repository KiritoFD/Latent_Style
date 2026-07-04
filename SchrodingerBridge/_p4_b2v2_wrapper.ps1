# FC-SB Phase 4 B2 V2 inference ablation wrapper
# Sets environment variables for B2 V2 checkpoint and runs _p4_infer_ablation.py
$env:P4_CKPT_PATH = "I:/Github/Latent_Style/SchrodingerBridge/exp/620_spectral_v2_weights/epoch_0001.pt"
$env:P4_CONFIG_PATH = "I:/Github/Latent_Style/SchrodingerBridge/configs/620_spectral_v2_weights.json"
$env:P4_BASELINE_CLIP = "0.6731"
$env:P4_BASELINE_LPIPS = "0.2781"
Set-Location I:/Github/Latent_Style/SchrodingerBridge
& "C:/Program Files/Python312/python.exe" $args
