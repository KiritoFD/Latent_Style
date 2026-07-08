$ErrorActionPreference = 'Continue'

Write-Host "=== P256 weave (latent256_e10/images) file count and extensions ==="
$ssh_out = ssh -p 2222 -o LogLevel=ERROR administrator@100.115.18.62 "dir /B I:\exp_our_models_eval\latent256_e10\images"
Write-Host $ssh_out

Write-Host ""
Write-Host "=== Count by extension ==="
$ssh_out = ssh -p 2222 -o LogLevel=ERROR administrator@100.115.18.62 "dir /B I:\exp_our_models_eval\latent256_e10\images\*.png 2>nul | find /C /V """
Write-Host "PNG: $ssh_out"
$ssh_out = ssh -p 2222 -o LogLevel=ERROR administrator@100.115.18.62 "dir /B I:\exp_our_models_eval\latent256_e10\images\*.jpg 2>nul | find /C /V """
Write-Host "JPG: $ssh_out"

Write-Host ""
Write-Host "=== exp_our_models_eval/pixel256_e3/images ==="
$ssh_out = ssh -p 2222 -o LogLevel=ERROR administrator@100.115.18.62 "dir /B I:\exp_our_models_eval\pixel256_e3 2>nul"
Write-Host $ssh_out
