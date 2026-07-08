$ErrorActionPreference = 'Continue'

Write-Host "=== latent256_b16_e10/full_eval/epoch_0010 ==="
$ssh_out = ssh -p 2222 -o LogLevel=ERROR administrator@100.115.18.62 "dir /B I:\Github\Latent_Style\SchrodingerBridge\exp\latent256_photo2art\latent256_b16_e10\full_eval\epoch_0010"
Write-Host $ssh_out

Write-Host ""
Write-Host "=== Count files in latent256_e10/images (D5-512 style) ==="
$ssh_out = ssh -p 2222 -o LogLevel=ERROR administrator@100.115.18.62 "powershell -NoProfile -Command (Get-ChildItem 'I:\exp_our_models_eval\latent256_e10\images' -File).Count"
Write-Host "Total files: $ssh_out"

Write-Host ""
Write-Host "=== Check extra_metrics.json for image_dir hint ==="
$ssh_out = ssh -p 2222 -o LogLevel=ERROR administrator@100.115.18.62 "type I:\exp_our_models_eval\latent256_e10\extra_metrics.json"
Write-Host $ssh_out

Write-Host ""
Write-Host "=== Check methods_extra.json ==="
$ssh_out = ssh -p 2222 -o LogLevel=ERROR administrator@100.115.18.62 "type I:\exp_our_models_eval\latent256_e10\methods_extra.json"
Write-Host $ssh_out
