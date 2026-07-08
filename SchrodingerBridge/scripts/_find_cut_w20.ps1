$ErrorActionPreference = 'Continue'

Write-Host "=== _eval_cut_w20.json ==="
$ssh_out = ssh -p 2222 -o LogLevel=ERROR administrator@100.115.18.62 "type I:\Github\Latent_Style\SchrodingerBridge\exp\_eval_cut_w20.json"
Write-Host $ssh_out

Write-Host ""
Write-Host "=== _eval_cut_d5_musiq.json ==="
$ssh_out = ssh -p 2222 -o LogLevel=ERROR administrator@100.115.18.62 "type I:\Github\Latent_Style\SchrodingerBridge\exp\_eval_cut_d5_musiq.json"
Write-Host $ssh_out

Write-Host ""
Write-Host "=== _eval_cut_w20_musiq.json ==="
$ssh_out = ssh -p 2222 -o LogLevel=ERROR administrator@100.115.18.62 "type I:\Github\Latent_Style\SchrodingerBridge\exp\_eval_cut_w20_musiq.json"
Write-Host $ssh_out
