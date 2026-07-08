$ErrorActionPreference = 'Continue'

Write-Host "=== Sample R5-WikiArt/identity/metrics.csv (first 3 lines) ==="
$ssh_out = ssh -p 2222 -o LogLevel=ERROR administrator@100.115.18.62 "powershell -NoProfile -Command Get-Content 'I:\Github\Latent_Style\SchrodingerBridge\exp\baseline_wikiarts20\identity\metrics.csv' -TotalCount 3"
Write-Host $ssh_out

Write-Host ""
Write-Host "=== Sample WEAVE D5 metrics.csv (first 3 lines) ==="
$ssh_out = ssh -p 2222 -o LogLevel=ERROR administrator@100.115.18.62 "powershell -NoProfile -Command Get-Content 'I:\Github\Latent_Style\SchrodingerBridge\exp\clean_base_v2\full_eval\epoch_0010\metrics.csv' -TotalCount 3"
Write-Host $ssh_out

Write-Host ""
Write-Host "=== Sample Seedream D5 metrics.csv (first 3 lines) ==="
$ssh_out = ssh -p 2222 -o LogLevel=ERROR administrator@100.115.18.62 "powershell -NoProfile -Command Get-Content 'I:\Github\Latent_Style\exp_baselines\seedream45_api\distinct5_512_seedream45_windhub_20260607_repaired750\metrics.csv' -TotalCount 3"
Write-Host $ssh_out

Write-Host ""
Write-Host "=== Sample P256 identity_256 (check if metrics.csv exists anywhere) ==="
$ssh_out = ssh -p 2222 -o LogLevel=ERROR administrator@100.115.18.62 "dir /B /S I:\exp_256_photo2art\identity_256\*.csv 2>nul"
Write-Host $ssh_out

Write-Host ""
Write-Host "=== Sample P256 identity_256 images filenames ==="
$ssh_out = ssh -p 2222 -o LogLevel=ERROR administrator@100.115.18.62 "dir /B I:\exp_256_photo2art\identity_256\images 2>nul | findstr /R \".jpg\" | head -5"
Write-Host $ssh_out

Write-Host ""
Write-Host "=== R5-WikiArt/identity images filenames ==="
$ssh_out = ssh -p 2222 -o LogLevel=ERROR administrator@100.115.18.62 "dir /B I:\Github\Latent_Style\SchrodingerBridge\exp\baseline_wikiarts20\identity\images 2>nul | head -5"
Write-Host $ssh_out
