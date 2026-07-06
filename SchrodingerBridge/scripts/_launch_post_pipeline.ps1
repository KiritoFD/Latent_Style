# Launch post_pipeline.ps1 as a SYSTEM scheduled task (survives SSH disconnects)
$taskName = "post_pipeline_fill"
$psScript = "I:\Github\Latent_Style\SchrodingerBridge\scripts\_post_pipeline.ps1"

# Remove existing task if any
schtasks /Delete /TN $taskName /F 2>$null

# Create one-time task running as SYSTEM
$cmd = "powershell.exe -NoProfile -ExecutionPolicy Bypass -File `"$psScript`""
schtasks /Create /TN $taskName /TR $cmd /SC ONCE /ST 23:59 /RU SYSTEM /RL HIGHEST /F

# Run it now
schtasks /Run /TN $taskName

Write-Host "Task $taskName created and started."
Start-Sleep -Seconds 3
schtasks /Query /TN $taskName /FO LIST
