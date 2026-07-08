$ErrorActionPreference = 'Continue'

Write-Host "=== Check D5-512 reference images (target_root) ==="
$ssh_out = ssh -p 2222 -o LogLevel=ERROR administrator@100.115.18.62 "dir /B I:\Github\Latent_Style\Dataset\distinct5_512\test"
Write-Host $ssh_out

Write-Host ""
Write-Host "=== Check R5-WikiArt reference images ==="
$ssh_out = ssh -p 2222 -o LogLevel=ERROR administrator@100.115.18.62 "dir /B I:\wikiart_distinct5_samam_512_classview\test 2>nul"
Write-Host $ssh_out

Write-Host ""
Write-Host "=== Check P256 reference images (wikiarts15_256) ==="
$ssh_out = ssh -p 2222 -o LogLevel=ERROR administrator@100.115.18.62 "dir /B /AD I:\wikiarts15_256 2>nul"
Write-Host $ssh_out

Write-Host ""
Write-Host "=== Sample P256 identity_256 image filenames ==="
$ssh_out = ssh -p 2222 -o LogLevel=ERROR administrator@100.115.18.62 "dir /B I:\exp_256_photo2art\identity_256\images 2>nul"
Write-Host ($ssh_out | Select-Object -First 5)

Write-Host ""
Write-Host "=== Sample D5-512 identity image filenames ==="
$ssh_out = ssh -p 2222 -o LogLevel=ERROR administrator@100.115.18.62 "dir /B I:\Github\Latent_Style\SchrodingerBridge\exp\baseline_v2\images\identity 2>nul"
Write-Host ($ssh_out | Select-Object -First 5)
