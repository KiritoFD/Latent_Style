$ErrorActionPreference = 'Continue'

# Step 1: Delete wrong 444 images in P256/weave
$wrong_dir = "g:\GitHub\Latent_Style\SchrodingerBridge\results\P256\weave"
Write-Host "=== Deleting wrong images in $wrong_dir ==="
$before = (Get-ChildItem $wrong_dir -File -ErrorAction SilentlyContinue).Count
Write-Host "Before: $before files"
Get-ChildItem $wrong_dir -File | Remove-Item -Force
$after = (Get-ChildItem $wrong_dir -File -ErrorAction SilentlyContinue).Count
Write-Host "After: $after files"

# Step 2: Check correct WEAVE P256 path on remote
Write-Host ""
Write-Host "=== Correct WEAVE P256 path: latent256_b16_e10/full_eval/epoch_0010/images ==="
$ssh_out = ssh -p 2222 -o LogLevel=ERROR administrator@100.115.18.62 "powershell -NoProfile -Command (Get-ChildItem 'I:\Github\Latent_Style\SchrodingerBridge\exp\latent256_photo2art\latent256_b16_e10\full_eval\epoch_0010\images' -File).Count"
Write-Host "File count: $ssh_out"

Write-Host ""
Write-Host "=== Sample filenames ==="
$ssh_out = ssh -p 2222 -o LogLevel=ERROR administrator@100.115.18.62 "dir /B I:\Github\Latent_Style\SchrodingerBridge\exp\latent256_photo2art\latent256_b16_e10\full_eval\epoch_0010\images 2>nul | findstr /R \"^cezanne ^Hayao ^monet ^photo ^vangogh\" | head -10"
Write-Host $ssh_out

# Step 3: Re-download from correct path
Write-Host ""
Write-Host "=== Re-downloading WEAVE P256 from correct path ==="
$t0 = Get-Date
& scp -P 2222 -o LogLevel=ERROR "administrator@100.115.18.62:I:/Github/Latent_Style/SchrodingerBridge/exp/latent256_photo2art/latent256_b16_e10/full_eval/epoch_0010/images/*.png" "$wrong_dir/"
$exit = $LASTEXITCODE
$elapsed = ((Get-Date) - $t0).TotalSeconds
$cnt = (Get-ChildItem $wrong_dir -Filter *.png -ErrorAction SilentlyContinue).Count
Write-Host "Result: exit=$exit, images=$cnt, time=$([math]::Round($elapsed, 1))s"
