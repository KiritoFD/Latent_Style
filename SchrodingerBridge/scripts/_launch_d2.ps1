# Launch D2 training as detached process
$script = "I:\Github\Latent_Style\SchrodingerBridge\scripts\_run_train_eval.ps1"
$proc = Start-Process -FilePath "powershell.exe" `
    -ArgumentList @("-NoProfile", "-ExecutionPolicy", "Bypass", "-File", $script, "d2_moment_hf1_15ep") `
    -WindowStyle Hidden -PassThru
Write-Output "Launched PID=$($proc.Id)"
$proc.Id | Out-File "C:\Users\Administrator\logs\d2_pid.txt"
