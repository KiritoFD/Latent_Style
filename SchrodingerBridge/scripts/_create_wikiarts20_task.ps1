# Create a schtasks job to run the wikiarts-20 pipeline (survives ssh disconnect)
# Runs immediately as SYSTEM, deletes itself after completion.

$taskName = "wikiarts20_all"
$script = "I:\Github\Latent_Style\SchrodingerBridge\scripts\_run_wikiarts20_all.ps1"

# Build the schtasks command: start now, run as SYSTEM, delete after done
$cmd = "powershell -ExecutionPolicy Bypass -NoProfile -File `"$script`""

Write-Output "Creating schtasks job: $taskName"
Write-Output "  Command: $cmd"

schtasks /Create /TN $taskName /TR $cmd /SC ONCE /ST 00:00 /RL HIGHEST /F 2>&1
schtasks /Run /TN $taskName 2>&1

Start-Sleep -Seconds 3
Write-Output ""
Write-Output "=== Task status ==="
schtasks /Query /TN $taskName /FO LIST 2>&1 | Select-String -Pattern "Status|LastRun|LastResult"
