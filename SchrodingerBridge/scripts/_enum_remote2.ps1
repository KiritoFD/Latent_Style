$ErrorActionPreference = 'Continue'

Write-Host "=== P256 method directories ==="
$ssh_out = ssh -p 2222 -o LogLevel=ERROR administrator@100.115.18.62 "dir /B I:\Github\Latent_Style\SchrodingerBridge\exp\baseline_wikiarts15_256"
Write-Host $ssh_out

Write-Host ""
Write-Host "=== R5-WikiArt method directories ==="
$ssh_out = ssh -p 2222 -o LogLevel=ERROR administrator@100.115.18.62 "dir /B I:\Github\Latent_Style\SchrodingerBridge\exp\baseline_wikiarts20"
Write-Host $ssh_out

Write-Host ""
Write-Host "=== D5-512 method directories ==="
$ssh_out = ssh -p 2222 -o LogLevel=ERROR administrator@100.115.18.62 "dir /B I:\Github\Latent_Style\SchrodingerBridge\exp\baseline_v2\images"
Write-Host $ssh_out

Write-Host ""
Write-Host "=== Checking WEAVE P256 (clean_base_v2 / wikiarts15_256) ==="
$ssh_out = ssh -p 2222 -o LogLevel=ERROR administrator@100.115.18.62 "dir /B I:\Github\Latent_Style\SchrodingerBridge\exp"
Write-Host $ssh_out
