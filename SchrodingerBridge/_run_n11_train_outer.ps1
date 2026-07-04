Start-Process -FilePath "powershell" -ArgumentList "-ExecutionPolicy","Bypass","-File","I:\Github\Latent_Style\SchrodingerBridge\_run_n11_train_inner.ps1" -WindowStyle Hidden
Write-Output "OUTER_STARTED_INDEPENDENT"
