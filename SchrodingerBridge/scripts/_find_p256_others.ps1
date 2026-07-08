$ErrorActionPreference = 'Continue'

Write-Host "=== Searching for Seedream P256 ==="
$ssh_out = ssh -p 2222 -o LogLevel=ERROR administrator@100.115.18.62 "dir /B /S I:\Github\Latent_Style\exp_baselines\seedream45_api 2>nul | findstr /I photo2art"
Write-Host $ssh_out

Write-Host ""
Write-Host "=== Searching for WEAVE P256 (wikiarts15_256) ==="
$ssh_out = ssh -p 2222 -o LogLevel=ERROR administrator@100.115.18.62 "dir /B I:\exp_256_photo2art"
Write-Host $ssh_out

Write-Host ""
Write-Host "=== Searching for clean_base_v2/wikiarts15_256 ==="
$ssh_out = ssh -p 2222 -o LogLevel=ERROR administrator@100.115.18.62 "dir /B I:\Github\Latent_Style\SchrodingerBridge\exp\clean_base_v2"
Write-Host $ssh_out

Write-Host ""
Write-Host "=== Looking for cut_256 ==="
$ssh_out = ssh -p 2222 -o LogLevel=ERROR administrator@100.115.18.62 "if exist I:\exp_256_photo2art\cut_256 (dir /B I:\exp_256_photo2art\cut_256) else (echo NOT FOUND)"
Write-Host $ssh_out
