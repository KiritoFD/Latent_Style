# Direct launcher - starts python training in background
Set-Location "I:\Github\Latent_Style\SchrodingerBridge"
$env:PYTHONPATH = "."
$logFile = "C:\Users\Administrator\logs\d6_style_consist_15ep_train_eval.out"

# Start training as a detached background process
$proc = Start-Process -FilePath "python.exe" `
    -ArgumentList @("-u", "src\run.py", "--config", "configs\d6_style_consist_15ep.json") `
    -WorkingDirectory "I:\Github\Latent_Style\SchrodingerBridge" `
    -RedirectStandardOutput $logFile `
    -RedirectStandardError "C:\Users\Administrator\logs\d6_style_consist_15ep_train_eval.err" `
    -WindowStyle Hidden `
    -PassThru

Write-Output "D6 training started, PID=$($proc.Id), log=$logFile"
