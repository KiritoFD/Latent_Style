# Use schtasks to launch master_pipeline persistently (SYSTEM account)
$ErrorActionPreference = "Continue"

$taskName = "master_pipeline"
$script = "I:\Github\Latent_Style\SchrodingerBridge\scripts\_master_pipeline.ps1"

# Delete existing task
schtasks /Delete /TN $taskName /F 2>$null | Out-Null

# Create task with proper date format (yyyy/mm/dd)
$startDate = Get-Date -Format "yyyy/MM/dd"
schtasks /Create /TN $taskName `
    /TR "powershell.exe -NoProfile -ExecutionPolicy Bypass -File `"$script`"" `
    /SC ONCE /ST 23:59 /SD $startDate `
    /RU SYSTEM /RL HIGHEST /F

Write-Host "Task created. Triggering..."
schtasks /Run /TN $taskName

Start-Sleep -Seconds 10

Write-Host "=== Status ==="
schtasks /Query /TN $taskName /FO LIST | Select-Object -First 8

Write-Host ""
Write-Host "=== Running processes ==="
Get-CimInstance Win32_Process -Filter "Name='powershell.exe' OR Name='python.exe'" |
    Select-Object ProcessId, Name, @{N='Start';E={$_.CreationDate}} |
    Format-Table -Auto

Write-Host ""
Write-Host "=== master_pipeline.log tail ==="
$mlog = "I:\Github\Latent_Style\SchrodingerBridge\logs\master_pipeline.log"
if (Test-Path $mlog) { Get-Content $mlog -Tail 10 }
