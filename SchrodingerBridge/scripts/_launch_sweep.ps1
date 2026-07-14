# Remote launcher: starts inference sweep in background
$logFile = "C:\Users\Administrator\logs\sweep_launcher.log"
$script = "I:\Github\Latent_Style\SchrodingerBridge\scripts\_run_inference_sweep.ps1"
Start-Process -FilePath "powershell.exe" -ArgumentList "-ExecutionPolicy","Bypass","-File",$script -WindowStyle Hidden -RedirectStandardOutput $logFile
Write-Output "SWEEP_LAUNCHED"
