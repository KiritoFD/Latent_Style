# Stop SaMam W20 v2 and master_pipeline
$ErrorActionPreference = "Continue"

Write-Host "=== Stopping python processes ==="
Get-CimInstance Win32_Process -Filter "Name='python.exe'" |
    Where-Object { $_.CommandLine -like "*_gen_samam*" -or $_.CommandLine -like "*_eval_unified*" } |
    ForEach-Object {
        Write-Host "Killing PID $($_.ProcessId): $($_.CommandLine.Substring(0,[Math]::Min(80,$_.CommandLine.Length)))"
        Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue
    }

Write-Host ""
Write-Host "=== Stopping master_pipeline powershell ==="
Get-CimInstance Win32_Process -Filter "Name='powershell.exe'" |
    Where-Object { $_.CommandLine -like "*_master_pipeline*" -or $_.CommandLine -like "*_eval_all_unified*" } |
    ForEach-Object {
        Write-Host "Killing PID $($_.ProcessId)"
        Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue
    }

Write-Host ""
Write-Host "=== Disabling schtasks ==="
schtasks /End /TN "master_pipeline" 2>$null
schtasks /Change /TN "master_pipeline" /Disable 2>$null
schtasks /Delete /TN "master_pipeline" /F 2>$null

Start-Sleep -Seconds 3

Write-Host ""
Write-Host "=== VRAM after cleanup ==="
nvidia-smi --query-gpu=memory.used,memory.free,memory.total --format=csv

Write-Host ""
Write-Host "=== Remaining processes ==="
Get-CimInstance Win32_Process -Filter "Name='python.exe' OR Name='powershell.exe'" |
    Select-Object ProcessId, Name | Format-Table -Auto

Write-Host ""
Write-Host "=== SaMam W20 images count ==="
$imgDir = "I:\Github\Latent_Style\SchrodingerBridge\exp\baseline_wikiarts20\samam\images"
if (Test-Path $imgDir) {
    $cnt = (Get-ChildItem $imgDir -File).Count
    Write-Host "samam_w20 images: $cnt / 750"
}
