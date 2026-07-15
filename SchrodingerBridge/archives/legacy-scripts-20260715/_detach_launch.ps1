# Launches the eval .ps1 as a fully detached process via Start-Process
$script = "I:\Github\Latent_Style\SchrodingerBridge\scripts\_run_ll_fm_eval.ps1"
$proc = Start-Process -FilePath "powershell.exe" `
    -ArgumentList @("-NoProfile", "-ExecutionPolicy", "Bypass", "-File", $script) `
    -WindowStyle Hidden -PassThru
Write-Output "Launched PID=$($proc.Id)"
