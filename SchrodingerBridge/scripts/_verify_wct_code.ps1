$ErrorActionPreference = "SilentlyContinue"
Write-Output "=== WCT eigh lines in remote spectral_bridge620.py ==="
Select-String -Path "I:\Github\Latent_Style\SchrodingerBridge\src\spectral_bridge620.py" -Pattern "eigh" | Select-Object LineNumber,Line | Format-Table -AutoSize -Wrap
Write-Output "=== DONE ==="
