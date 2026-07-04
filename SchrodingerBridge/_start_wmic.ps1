$ErrorActionPreference = "Continue"
Set-Location "I:\Github\Latent_Style\SchrodingerBridge"
Remove-Item -Path "train_stage1_log.txt" -Force -ErrorAction SilentlyContinue
Remove-Item -Path "train_stage1_err.txt" -Force -ErrorAction SilentlyContinue

# Use WMIC to create a truly independent process (survives SSH disconnect)
$cmd = 'python -u -X faulthandler "I:\Github\Latent_Style\SchrodingerBridge\src\run.py" --config "I:\Github\Latent_Style\SchrodingerBridge\configs\clean_base_v2.json" 1>"I:\Github\Latent_Style\SchrodingerBridge\train_stage1_log.txt" 2>"I:\Github\Latent_Style\SchrodingerBridge\train_stage1_err.txt"'
wmic process call create "$cmd"
Write-Output "Started at $(Get-Date)"
Start-Sleep -Seconds 10
Write-Output "=== Process status ==="
tasklist | findstr python
Write-Output ""
Write-Output "=== ERR (last 10 lines) ==="
Get-Content "train_stage1_err.txt" -ErrorAction SilentlyContinue -Tail 10
