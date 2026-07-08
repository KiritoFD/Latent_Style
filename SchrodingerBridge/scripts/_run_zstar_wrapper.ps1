# Z-STAR wrapper with full logging (fp16 + skip_null_opt)
$logFile = "C:\Users\Administrator\logs\zstar_run.log"
$null = New-Item -ItemType Directory -Path "C:\Users\Administrator\logs" -Force

"[$(Get-Date)] Starting Z-STAR inference (fp16, skip_null_opt)..." | Out-File -FilePath $logFile -Encoding utf8

$proc = Start-Process -FilePath "C:\Program Files\Python312\python.exe" -ArgumentList "C:\Users\Administrator\_run_zstar_remote.py --fp16 --skip_null_opt" -WorkingDirectory "C:\Users\Administrator" -RedirectStandardOutput "C:\Users\Administrator\logs\zstar_stdout.log" -RedirectStandardError "C:\Users\Administrator\logs\zstar_stderr.log" -NoNewWindow -Wait -PassThru

"[$(Get-Date)] Z-STAR exited with code: $($proc.ExitCode)" | Out-File -FilePath $logFile -Encoding utf8 -Append
