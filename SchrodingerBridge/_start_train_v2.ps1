$ErrorActionPreference = "Continue"
Set-Location "I:\Github\Latent_Style\SchrodingerBridge"
Remove-Item -Path "train_stage1_log.txt" -Force -ErrorAction SilentlyContinue
Remove-Item -Path "train_stage1_err.txt" -Force -ErrorAction SilentlyContinue
$proc = Start-Process -FilePath "python.exe" `
    -ArgumentList "-u","src\run.py","--config","configs\clean_base_v2.json" `
    -RedirectStandardOutput "train_stage1_log.txt" `
    -RedirectStandardError "train_stage1_err.txt" `
    -NoNewWindow -PassThru
Write-Output "Started PID=$($proc.Id)"
Start-Sleep -Seconds 5
Get-Content "train_stage1_log.txt" -ErrorAction SilentlyContinue
Write-Output "---ERR---"
Get-Content "train_stage1_err.txt" -ErrorAction SilentlyContinue
