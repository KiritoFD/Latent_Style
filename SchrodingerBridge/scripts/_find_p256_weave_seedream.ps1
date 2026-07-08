$ErrorActionPreference = 'Continue'

Write-Host "=== eval_ours_latent256_e10.json (WEAVE P256) ==="
$ssh_out = ssh -p 2222 -o LogLevel=ERROR administrator@100.115.18.62 "type I:\exp_256_photo2art\eval_ours_latent256_e10.json"
Write-Host $ssh_out

Write-Host ""
Write-Host "=== Search for seedream + 256 in exp_baselines ==="
$ssh_out = ssh -p 2222 -o LogLevel=ERROR administrator@100.115.18.62 "dir /B /AD I:\Github\Latent_Style\exp_baselines\seedream45_api"
Write-Host $ssh_out

Write-Host ""
Write-Host "=== Search for seedream_256 images anywhere ==="
$ssh_out = ssh -p 2222 -o LogLevel=ERROR administrator@100.115.18.62 "dir /B /S /AD I:\exp_256_photo2art 2>nul | findstr /I seedream"
Write-Host $ssh_out

Write-Host ""
Write-Host "=== Search I:\ root for seedream_256 ==="
$ssh_out = ssh -p 2222 -o LogLevel=ERROR administrator@100.115.18.62 "dir /B /AD I:\ 2>nul"
Write-Host $ssh_out
