$ErrorActionPreference = 'Continue'

Write-Host "=== Find Python on remote ==="
$ssh_out = ssh -p 2222 -o LogLevel=ERROR administrator@100.115.18.62 "where python"
Write-Host "where python: $ssh_out"

Write-Host ""
Write-Host "=== Find conda envs ==="
$ssh_out = ssh -p 2222 -o LogLevel=ERROR administrator@100.115.18.62 "C:\Users\Administrator\miniconda3\condabin\conda.bat env list 2>nul"
Write-Host "conda envs: $ssh_out"

Write-Host ""
Write-Host "=== D5 test dir (try different paths) ==="
foreach ($p in @(
    "I:\Github\Latent_Style\Dataset\distinct5_512\test",
    "I:\datasets\distinct5_512\test",
    "I:\wikiart_distinct5_samam_512_classview\test"
)) {
    $exists = ssh -p 2222 -o LogLevel=ERROR administrator@100.115.18.62 "if exist `"$p`" (echo EXISTS) else (echo MISSING)"
    Write-Host "${p}: $exists"
}

Write-Host ""
Write-Host "=== P2A test dir styles ==="
$ssh_out = ssh -p 2222 -o LogLevel=ERROR administrator@100.115.18.62 "dir /B /AD I:\datasets\legacy256_overfit50\test"
Write-Host "P2A styles: $ssh_out"

Write-Host ""
Write-Host "=== R5 test dir styles ==="
$ssh_out = ssh -p 2222 -o LogLevel=ERROR administrator@100.115.18.62 "dir /B /AD I:\datasets\wikiarts20_512_test"
Write-Host "R5 styles: $ssh_out"
