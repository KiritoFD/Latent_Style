$ErrorActionPreference = "Stop"

$taskName = "Codex_VAE_Backend_StatusWatch"
$repoRoot = "G:\GitHub\Latent_Style\SchrodingerBridge"
$scriptPath = Join-Path $repoRoot "tools\experiments\check_vae_backend_remote_status.ps1"
$action = "powershell.exe -NoProfile -ExecutionPolicy Bypass -File `"$scriptPath`""

$ErrorActionPreference = "Continue"
schtasks /delete /tn $taskName /f 2>$null | Out-Null
$ErrorActionPreference = "Stop"

schtasks /create /tn $taskName /tr $action /sc minute /mo 10 /f | Out-Host
schtasks /run /tn $taskName | Out-Host
Start-Sleep -Seconds 5

Write-Host "---TASK---"
schtasks /query /tn $taskName /fo LIST /v
Write-Host "---STATUS---"
$status = Join-Path $repoRoot "exp\vae_backend_256_status\status.md"
if (Test-Path $status) {
    Get-Content $status -TotalCount 100
} else {
    Write-Host "status.md not written yet"
}
