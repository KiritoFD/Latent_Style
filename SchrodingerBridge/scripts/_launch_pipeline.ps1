# Launch the pipeline as a background scheduled task (SYSTEM, 4hr timeout)
$taskName = "pipeline_fill_main"
$script = "I:\Github\Latent_Style\SchrodingerBridge\scripts\_pipeline_fill_main.ps1"

# Delete existing task if any
schtasks /Delete /TN $taskName /F 2>$null

# Create new task
schtasks /Create /TN $taskName /TR "powershell.exe -NoProfile -ExecutionPolicy Bypass -File $script" /SC ONCE /ST 23:59 /RU SYSTEM /F

# Run it now
schtasks /Run /TN $taskName

Write-Host "Task '$taskName' created and started."
Write-Host "Monitor with: schtasks /Query /TN $taskName /FO LIST"
Write-Host "Log: I:\Github\Latent_Style\SchrodingerBridge\logs\pipeline_fill_main.log"
