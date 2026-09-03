$ErrorActionPreference = "Continue"
Set-Location "I:\Github\Latent_Style\SchrodingerBridge"
$taskName = "LANCET_VAE_Backend_256_SDXL"
schtasks /delete /tn $taskName /f 2>$null | Out-Null
schtasks /create /tn $taskName /tr "I:\Github\Latent_Style\SchrodingerBridge\start_remote_vae_sdxl_smoke.bat" /sc once /st 00:00 /f | Out-Host
schtasks /run /tn $taskName | Out-Host
Start-Sleep -Seconds 5
schtasks /query /tn $taskName /fo LIST /v
