# Launcher for D6 training - starts in background
Set-Location "I:\Github\Latent_Style\SchrodingerBridge"
$env:PYTHONPATH = "."
Start-Process -FilePath "powershell.exe" `
    -ArgumentList @("-NoProfile", "-ExecutionPolicy", "Bypass", "-File", "I:\Github\Latent_Style\SchrodingerBridge\scripts\_run_train_eval.ps1", "d6_style_consist_15ep") `
    -WorkingDirectory "I:\Github\Latent_Style\SchrodingerBridge" `
    -WindowStyle Hidden
Write-Output "D6 training launched in background"
