$ErrorActionPreference = 'Continue'

Write-Host "=== Search for distinct5 anywhere on I: ==="
$ssh_out = ssh -p 2222 -o LogLevel=ERROR administrator@100.115.18.62 "dir /B /S /AD I:\ 2>nul | findstr /I distinct5"
Write-Host $ssh_out

Write-Host ""
Write-Host "=== Search on G: ==="
$ssh_out = ssh -p 2222 -o LogLevel=ERROR administrator@100.115.18.62 "dir /B /S /AD G:\Github\Latent_Style\Dataset\distinct5 2>nul"
Write-Host $ssh_out

Write-Host ""
Write-Host "=== Check if G: drive exists ==="
$ssh_out = ssh -p 2222 -o LogLevel=ERROR administrator@100.115.18.62 "if exist G:\ (echo EXISTS) else (echo MISSING)"
Write-Host "G: $ssh_out"

Write-Host ""
Write-Host "=== Check existing R5 baseline images dir ==="
$ssh_out = ssh -p 2222 -o LogLevel=ERROR administrator@100.115.18.62 "if exist I:\Github\Latent_Style\SchrodingerBridge\exp\baseline_wikiarts20\identity\images (dir /B I:\Github\Latent_Style\SchrodingerBridge\exp\baseline_wikiarts20\identity\images 2>nul | head -3)"
Write-Host "R5 identity sample: $ssh_out"

Write-Host ""
Write-Host "=== Check what R5 styles were actually used (from image filenames) ==="
$ssh_out = ssh -p 2222 -o LogLevel=ERROR administrator@100.115.18.62 "dir /B I:\Github\Latent_Style\SchrodingerBridge\exp\baseline_wikiarts20\identity\images 2>nul"
$first3 = $ssh_out -split "`n" | Select-Object -First 3
Write-Host ($first3 -join "`n")

Write-Host ""
Write-Host "=== Python version check ==="
$ssh_out = ssh -p 2222 -o LogLevel=ERROR administrator@100.115.18.42 "C:\Program Files\Python312\python.exe --version 2>&1"
Write-Host $ssh_out

Write-Host ""
Write-Host "=== Check pip/diffusers availability ==="
$ssh_out = ssh -p 2222 -o LogLevel=ERROR administrator@100.115.18.42 "C:\Program Files\Python312\python.exe -c `"import diffusers; print(diffusers.__version__)`" 2>&1"
Write-Host $ssh_out
