$ErrorActionPreference = "Continue"
Set-Location "I:\Github\Latent_Style\SchrodingerBridge"
Remove-Item -Path "train_stage1_log.txt" -Force -ErrorAction SilentlyContinue
Remove-Item -Path "train_stage1_err.txt" -Force -ErrorAction SilentlyContinue

# Delete old task if exists
schtasks /Delete /TN "sb_train_stage1" /F 2>$null | Out-Null

# Create a one-time scheduled task that runs immediately
$pythonExe = (Get-Command python.exe).Source
$workDir = "I:\Github\Latent_Style\SchrodingerBridge"
$script = "`"$pythonExe`" -u -X faulthandler `"$workDir\src\run.py`" --config `"$workDir\configs\clean_base_v2.json`" 1> `"$workDir\train_stage1_log.txt`" 2> `"$workDir\train_stage1_err.txt`""

# Write the command to a batch file
$batPath = "$workDir\_run_train.bat"
Set-Content -Path $batPath -Value $script -Encoding ASCII

# Create scheduled task to run the batch file
schtasks /Create /TN "sb_train_stage1" /TR $batPath /SC ONCE /ST 00:00 /RL HIGHEST /F
schtasks /Run /TN "sb_train_stage1"

Write-Output "Task started at $(Get-Date)"
Start-Sleep -Seconds 8
Write-Output "=== Process status ==="
tasklist | findstr python
Write-Output ""
Write-Output "=== ERR (last 15 lines) ==="
Get-Content "train_stage1_err.txt" -ErrorAction SilentlyContinue -Tail 15
