$ErrorActionPreference = 'Continue'

Write-Host "=== latent256_photo2art/latent256_b16_e10 ==="
$ssh_out = ssh -p 2222 -o LogLevel=ERROR administrator@100.115.18.62 "dir /B I:\Github\Latent_Style\SchrodingerBridge\exp\latent256_photo2art\latent256_b16_e10"
Write-Host $ssh_out

Write-Host ""
Write-Host "=== exp_our_models_eval ==="
$ssh_out = ssh -p 2222 -o LogLevel=ERROR administrator@100.115.18.62 "dir /B I:\exp_our_models_eval"
Write-Host $ssh_out

Write-Host ""
Write-Host "=== Search for any seedream 256 json in exp/ ==="
$ssh_out = ssh -p 2222 -o LogLevel=ERROR administrator@100.115.18.62 "dir /B I:\Github\Latent_Style\SchrodingerBridge\exp\_eval_seedream*"
Write-Host $ssh_out

Write-Host ""
Write-Host "=== pixel256_photo2art contents (could be WEAVE P256) ==="
$ssh_out = ssh -p 2222 -o LogLevel=ERROR administrator@100.115.18.62 "dir /B I:\Github\Latent_Style\SchrodingerBridge\exp\pixel256_photo2art\pixel256_b2_e10_softmax"
Write-Host $ssh_out
