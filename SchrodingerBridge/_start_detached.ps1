$ErrorActionPreference = "Continue"
Set-Location "I:\Github\Latent_Style\SchrodingerBridge"
Remove-Item -Path "train_stage1_log.txt" -Force -ErrorAction SilentlyContinue
Remove-Item -Path "train_stage1_err.txt" -Force -ErrorAction SilentlyContinue

# Create a detached process using WMIC (truly independent of SSH session)
$cmdLine = 'python -u -X faulthandler "I:\Github\Latent_Style\SchrodingerBridge\src\run.py" --config "I:\Github\Latent_Style\SchrodingerBridge\configs\clean_base_v2.json"'
$proc = Start-Process -FilePath "python.exe" `
    -ArgumentList "-u","-X","faulthandler","src\run.py","--config","configs\clean_base_v2.json" `
    -RedirectStandardOutput "train_stage1_log.txt" `
    -RedirectStandardError "train_stage1_err.txt" `
    -WindowStyle Hidden `
    -PassThru

Write-Output "Started PID=$($proc.Id) at $(Get-Date)"
Start-Sleep -Seconds 3

# Verify process is running
$running = Get-Process -Id $proc.Id -ErrorAction SilentlyContinue
if ($running) {
    Write-Output "Process is running: PID=$($proc.Id) WS=$([math]::Round($running.WorkingSet64/1MB,1))MB"
} else {
    Write-Output "Process exited immediately! ExitCode=$($proc.ExitCode)"
    Write-Output "=== ERR ==="
    Get-Content "train_stage1_err.txt" -ErrorAction SilentlyContinue
    exit 1
}
