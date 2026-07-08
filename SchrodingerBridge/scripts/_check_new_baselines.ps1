$ErrorActionPreference = 'Continue'

Write-Host "=== 1. Check existing baseline directories ==="
$ssh_out = ssh -p 2222 -o LogLevel=ERROR administrator@100.115.18.62 "dir /B I:\Github\Latent_Style\SchrodingerBridge\exp\baseline_v2\images"
Write-Host "D5-512 methods: $ssh_out"

Write-Host ""
Write-Host "=== 2. Check P256 methods ==="
$ssh_out = ssh -p 2222 -o LogLevel=ERROR administrator@100.115.18.62 "dir /B I:\exp_256_photo2art"
Write-Host "P256 methods: $ssh_out"

Write-Host ""
Write-Host "=== 3. Check R5-WikiArt methods ==="
$ssh_out = ssh -p 2222 -o LogLevel=ERROR administrator@100.115.18.62 "dir /B I:\Github\Latent_Style\SchrodingerBridge\exp\baseline_wikiarts20"
Write-Host "R5 methods: $ssh_out"

Write-Host ""
Write-Host "=== 4. Search for StyleAligned ==="
$ssh_out = ssh -p 2222 -o LogLevel=ERROR administrator@100.115.18.62 "dir /B /S I:\Github\Latent_Style 2>nul | findstr /I stylealign"
Write-Host "StyleAligned: $ssh_out"

Write-Host ""
Write-Host "=== 5. Search for Z-STAR ==="
$ssh_out = ssh -p 2222 -o LogLevel=ERROR administrator@100.115.18.62 "dir /B /S I:\Github\Latent_Style 2>nul | findstr /I zstar z-star z_star"
Write-Host "Z-STAR: $ssh_out"

Write-Host ""
Write-Host "=== 6. Search for StyleShot ==="
$ssh_out = ssh -p 2222 -o LogLevel=ERROR administrator@100.115.18.62 "dir /B /S I:\Github\Latent_Style 2>nul | findstr /I styleshot style_shot"
Write-Host "StyleShot: $ssh_out"

Write-Host ""
Write-Host "=== 7. Check baseline_v2 eval JSONs ==="
$ssh_out = ssh -p 2222 -o LogLevel=ERROR administrator@100.115.18.62 "dir /B I:\Github\Latent_Style\SchrodingerBridge\exp\_eval_stylealign*"
Write-Host "StyleAligned eval: $ssh_out"

Write-Host ""
Write-Host "=== 8. Check for any stylealigned/zstar/styleshot code repos ==="
$ssh_out = ssh -p 2222 -o LogLevel=ERROR administrator@100.115.18.62 "dir /B C:\Users\Administrator\ 2>nul | findstr /I style"
Write-Host "Admin home style*: $ssh_out"
$ssh_out = ssh -p 2222 -o LogLevel=ERROR administrator@100.115.18.62 "dir /B C:\Users\Administrator\ 2>nul"
Write-Host "Admin home all: $ssh_out"
