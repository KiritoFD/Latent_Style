$ErrorActionPreference = 'Continue'

Write-Host "=== 1. D5-512 existing methods ==="
$ssh_out = ssh -p 2222 -o LogLevel=ERROR administrator@100.115.18.62 "dir /B I:\Github\Latent_Style\SchrodingerBridge\exp\baseline_v2\images"
Write-Host $ssh_out

Write-Host ""
Write-Host "=== 2. P256 existing methods ==="
$ssh_out = ssh -p 2222 -o LogLevel=ERROR administrator@100.115.18.62 "dir /B I:\exp_256_photo2art"
Write-Host $ssh_out

Write-Host ""
Write-Host "=== 3. R5-WikiArt existing methods ==="
$ssh_out = ssh -p 2222 -o LogLevel=ERROR administrator@100.115.18.62 "dir /B I:\Github\Latent_Style\SchrodingerBridge\exp\baseline_wikiarts20"
Write-Host $ssh_out

Write-Host ""
Write-Host "=== 4. Search StyleAligned anywhere ==="
$ssh_out = ssh -p 2222 -o LogLevel=ERROR administrator@100.115.18.62 "dir /B /S /AD C:\Users\Administrator\ 2>nul | findstr /I stylealign"
Write-Host "C: drive: $ssh_out"
$ssh_out = ssh -p 2222 -o LogLevel=ERROR administrator@100.115.18.62 "dir /B /S /AD I:\ 2>nul | findstr /I stylealign"
Write-Host "I: drive: $ssh_out"

Write-Host ""
Write-Host "=== 5. Search Z-STAR anywhere ==="
$ssh_out = ssh -p 2222 -o LogLevel=ERROR administrator@100.115.18.62 "dir /B /S /AD C:\Users\Administrator\ 2>nul | findstr /I zstar z-star"
Write-Host "C: zstar: $ssh_out"
$ssh_out = ssh -p 2222 -o LogLevel=ERROR administrator@100.115.18.62 "dir /B /S /AD I:\ 2>nul | findstr /I zstar z-star"
Write-Host "I: zstar: $ssh_out"

Write-Host ""
Write-Host "=== 6. Search StyleShot anywhere ==="
$ssh_out = ssh -p 2222 -o LogLevel=ERROR administrator@100.115.18.62 "dir /B /S /AD C:\Users\Administrator\ 2>nul | findstr /I styleshot"
Write-Host "C: styleshot: $ssh_out"
$ssh_out = ssh -p 2222 -o LogLevel=ERROR administrator@100.115.18.62 "dir /B /S /AD I:\ 2>nul | findstr /I styleshot"
Write-Host "I: styleshot: $ssh_out"

Write-Host ""
Write-Host "=== 7. Check baseline_v2 images stylealigned ==="
$ssh_out = ssh -p 2222 -o LogLevel=ERROR administrator@100.115.18.62 "if exist I:\Github\Latent_Style\SchrodingerBridge\exp\baseline_v2\images\stylealigned (echo EXISTS) else (echo MISSING)"
Write-Host "D5 stylealigned: $ssh_out"
$ssh_out = ssh -p 2222 -o LogLevel=ERROR administrator@100.115.18.62 "if exist I:\Github\Latent_Style\SchrodingerBridge\exp\baseline_v2\images\ipadapter (echo EXISTS) else (echo MISSING)"
Write-Host "D5 ipadapter: $ssh_out"
