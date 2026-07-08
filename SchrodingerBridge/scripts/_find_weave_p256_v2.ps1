$ErrorActionPreference = 'Continue'

Write-Host "=== WEAVE P256 full_eval ==="
$ssh_out = ssh -p 2222 -o LogLevel=ERROR administrator@100.115.18.62 "dir /B I:\Github\Latent_Style\SchrodingerBridge\exp\latent256_photo2art\latent256_b16_e10\full_eval"
Write-Host $ssh_out

Write-Host ""
Write-Host "=== exp_our_models_eval/latent256_e10 ==="
$ssh_out = ssh -p 2222 -o LogLevel=ERROR administrator@100.115.18.62 "dir /B I:\exp_our_models_eval\latent256_e10"
Write-Host $ssh_out

Write-Host ""
Write-Host "=== Search I: for any photo2art256 seedream dir ==="
$ssh_out = ssh -p 2222 -o LogLevel=ERROR administrator@100.115.18.62 "dir /B /AD /S I:\exp_256_photo2art 2>nul"
Write-Host $ssh_out
