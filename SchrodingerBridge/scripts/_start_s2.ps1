param([string]$ConfigName)
$logDir = "C:\Users\Administrator\logs"
if (-not (Test-Path $logDir)) { New-Item -ItemType Directory -Path $logDir -Force }
Start-Process -FilePath "powershell.exe" `
    -ArgumentList "-ExecutionPolicy","Bypass","-File","C:\Users\Administrator\_run_s2_train_eval.ps1","-ConfigName",$ConfigName `
    -WindowStyle Hidden `
    -RedirectStandardOutput "$logDir\s2_${ConfigName}_out.txt" `
    -RedirectStandardError "$logDir\s2_${ConfigName}_err.txt"
Write-Host "Started background process for $ConfigName"
