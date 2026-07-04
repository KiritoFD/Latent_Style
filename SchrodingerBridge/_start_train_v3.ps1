$ErrorActionPreference = "Continue"
Set-Location "I:\Github\Latent_Style\SchrodingerBridge"
Remove-Item -Path "train_stage1_log.txt" -Force -ErrorAction SilentlyContinue
Remove-Item -Path "train_stage1_err.txt" -Force -ErrorAction SilentlyContinue
# Use faulthandler to catch segfaults; -u for unbuffered stdout
$proc = Start-Process -FilePath "python.exe" `
    -ArgumentList "-u","-X","faulthandler","src\run.py","--config","configs\clean_base_v2.json" `
    -RedirectStandardOutput "train_stage1_log.txt" `
    -RedirectStandardError "train_stage1_err.txt" `
    -NoNewWindow -PassThru
Write-Output "Started PID=$($proc.Id)"
Start-Sleep -Seconds 15
Write-Output "=== LOG ==="
Get-Content "train_stage1_log.txt" -ErrorAction SilentlyContinue
Write-Output ""
Write-Output "=== ERR ==="
Get-Content "train_stage1_err.txt" -ErrorAction SilentlyContinue
Write-Output ""
Write-Output "=== Process running? ==="
$running = Get-Process -Id $proc.Id -ErrorAction SilentlyContinue
if ($running) {
    Write-Output "YES - PID=$($proc.Id) CPU=$($running.CPU)s WS=$([math]::Round($running.WorkingSet64/1MB,1))MB"
} else {
    Write-Output "NO - process exited with code $($proc.ExitCode)"
}
