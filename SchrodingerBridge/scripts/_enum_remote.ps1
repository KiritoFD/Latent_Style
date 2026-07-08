$ErrorActionPreference = 'Continue'

# Enumerate all P256 method directories on remote
Write-Host "=== Enumerating P256 method directories on remote ==="
$ssh_out = ssh -p 2222 -o LogLevel=ERROR administrator@100.115.18.62 "powershell -NoProfile -Command Get-ChildItem 'I:\Github\Latent_Style\SchrodingerBridge\exp\baseline_wikiarts15_256' -Directory | Select-Object -ExpandProperty Name"
Write-Host $ssh_out

Write-Host ""
Write-Host "=== Enumerating R5-WikiArt method directories on remote ==="
$ssh_out = ssh -p 2222 -o LogLevel=ERROR administrator@100.115.18.62 "powershell -NoProfile -Command Get-ChildItem 'I:\Github\Latent_Style\SchrodingerBridge\exp\baseline_wikiarts20' -Directory | Select-Object -ExpandProperty Name"
Write-Host $ssh_out

Write-Host ""
Write-Host "=== Checking D5-512 method directories ==="
$ssh_out = ssh -p 2222 -o LogLevel=ERROR administrator@100.115.18.62 "powershell -NoProfile -Command Get-ChildItem 'I:\Github\Latent_Style\SchrodingerBridge\exp\baseline_v2\images' -Directory | Select-Object -ExpandProperty Name"
Write-Host $ssh_out

Write-Host ""
Write-Host "=== Checking for WEAVE R5-WikiArt path ==="
$ssh_out = ssh -p 2222 -o LogLevel=ERROR administrator@100.115.18.62 "powershell -NoProfile -Command if (Test-Path 'I:\Github\Latent_Style\SchrodingerBridge\exp\wikiarts20_eval\images') { (Get-ChildItem 'I:\Github\Latent_Style\SchrodingerBridge\exp\wikiarts20_eval\images' -Filter *.png).Count } else { Write-Host 'NOT FOUND' }"
Write-Host $ssh_out
