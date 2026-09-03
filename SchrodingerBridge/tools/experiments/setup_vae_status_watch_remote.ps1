$ErrorActionPreference = "Stop"
Set-Location "I:\Github\Latent_Style\SchrodingerBridge"
New-Item -ItemType Directory -Force -Path "exp\vae_backend_256_status" | Out-Null

$taskName = "LANCET_VAE_Backend_StatusWatch"
$scriptPath = "I:\Github\Latent_Style\SchrodingerBridge\tools\experiments\watch_vae_backend_status.ps1"
$action = "powershell.exe -NoProfile -ExecutionPolicy Bypass -File `"$scriptPath`""

schtasks /delete /tn $taskName /f 2>$null | Out-Null
schtasks /create /tn $taskName /tr $action /sc minute /mo 10 /f | Out-Host
schtasks /run /tn $taskName | Out-Host
Start-Sleep -Seconds 3

Write-Host "---TASK---"
schtasks /query /tn $taskName /fo LIST /v
Write-Host "---STATUS---"
if (Test-Path "exp\vae_backend_256_status\status.md") {
    Get-Content "exp\vae_backend_256_status\status.md" -TotalCount 80
} else {
    Write-Host "status.md not written yet"
}
