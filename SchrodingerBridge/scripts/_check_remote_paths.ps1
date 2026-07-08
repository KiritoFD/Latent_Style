$ErrorActionPreference = 'Continue'

Write-Host "=== 1. P2A test dir candidates ==="
foreach ($p in @(
    "I:\datasets\legacy256_overfit50\test",
    "I:\Github\Latent_Style\Dataset\wikiarts15_256\test",
    "I:\wikiarts15_256\test",
    "I:\datasets\wikiarts15_256\test"
)) {
    $exists = ssh -p 2222 -o LogLevel=ERROR administrator@100.115.18.62 "if exist `"$p`" (echo EXISTS) else (echo MISSING)"
    Write-Host "${p}: $exists"
}

Write-Host ""
Write-Host "=== 2. R5 test dir candidates ==="
foreach ($p in @(
    "I:\datasets\wikiarts20_512_test",
    "I:\Github\Latent_Style\Dataset\wikiart_random20_512\test",
    "I:\wikiart_random20_512\test",
    "I:\datasets\wikiart_random20_512\test"
)) {
    $exists = ssh -p 2222 -o LogLevel=ERROR administrator@100.115.18.62 "if exist `"$p`" (echo EXISTS) else (echo MISSING)"
    Write-Host "${p}: $exists"
}

Write-Host ""
Write-Host "=== 3. D5 test dir ==="
$ssh_out = ssh -p 2222 -o LogLevel=ERROR administrator@100.115.18.62 "dir /B /AD I:\Github\Latent_Style\Dataset\distinct5_512\test"
Write-Host "D5 styles: $ssh_out"

Write-Host ""
Write-Host "=== 4. Check style_aligned upload ==="
$ssh_out = ssh -p 2222 -o LogLevel=ERROR administrator@100.115.18.62 "dir /B C:\Users\Administrator\style_aligned"
Write-Host "SA module files: $ssh_out"

Write-Host ""
Write-Host "=== 5. Check __init__.py ==="
$ssh_out = ssh -p 2222 -o LogLevel=ERROR administrator@100.115.18.62 "if exist C:\Users\Administrator\style_aligned\__init__.py (echo EXISTS) else (echo MISSING)"
Write-Host "__init__.py: $ssh_out"

Write-Host ""
Write-Host "=== 6. Python on remote - check diffusers version ==="
$ssh_out = ssh -p 2222 -o LogLevel=ERROR administrator@100.115.18.62 "C:\Users\Administrator\miniconda3\python.exe -c \"import diffusers; print(diffusers.__version__)\""
Write-Host "diffusers: $ssh_out"
