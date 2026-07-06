# Launch W20 full gen via schtasks (SYSTEM, persistent)
$ErrorActionPreference = "Continue"

$taskName = "w20_full_gen"
$script = "I:\Github\Latent_Style\SchrodingerBridge\scripts\_w20_full_gen.ps1"

# Delete existing task
schtasks /Delete /TN $taskName /F 2>$null | Out-Null

# Create task
$startDate = Get-Date -Format "yyyy/MM/dd"
schtasks /Create /TN $taskName `
    /TR "powershell.exe -NoProfile -ExecutionPolicy Bypass -File `"$script`"" `
    /SC ONCE /ST 23:59 /SD $startDate `
    /RU SYSTEM /RL HIGHEST /F

Write-Host "Task created. Triggering..."
schtasks /Run /TN $taskName

Start-Sleep -Seconds 10

Write-Host "=== Status ==="
schtasks /Query /TN $taskName /FO LIST | Select-Object -First 5

Write-Host ""
Write-Host "=== Running processes ==="
Get-CimInstance Win32_Process -Filter "Name='python.exe' OR Name='powershell.exe'" |
    Select-Object ProcessId, Name, @{N='Start';E={$_.CreationDate}} |
    Format-Table -Auto

Write-Host ""
Write-Host "=== VRAM ==="
nvidia-smi --query-gpu=memory.used,memory.free --format=csv

Write-Host ""
Write-Host "=== Log tail ==="
$log = "I:\Github\Latent_Style\SchrodingerBridge\logs\sdturbo_w20_full.log"
if (Test-Path $log) {
    Get-Content $log -Tail 10
}
