$ErrorActionPreference = 'Continue'

Write-Host "=== latent256_b16_e10/full_eval ==="
$ssh_out = ssh -p 2222 -o LogLevel=ERROR administrator@100.115.18.62 "dir /B I:\Github\Latent_Style\SchrodingerBridge\exp\latent256_photo2art\latent256_b16_e10\full_eval"
Write-Host $ssh_out

Write-Host ""
Write-Host "=== exp_our_models_eval/latent256_e10 ==="
$ssh_out = ssh -p 2222 -o LogLevel=ERROR administrator@100.115.18.62 "dir /B I:\exp_our_models_eval\latent256_e10"
Write-Host $ssh_out

Write-Host ""
Write-Host "=== Check Seedream P256 paper values source - search seedream*256 anywhere on I: ==="
$ssh_out = ssh -p 2222 -o LogLevel=ERROR administrator@100.115.18.62 "dir /B /S I:\Github\Latent_Style\exp_baselines 2>nul | findstr /I seedream"
Write-Host $ssh_out

Write-Host ""
Write-Host "=== exp_baselines listing ==="
$ssh_out = ssh -p 2222 -o LogLevel=ERROR administrator@100.115.18.62 "dir /B I:\Github\Latent_Style\exp_baselines"
Write-Host $ssh_out
