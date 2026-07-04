$queue = "I:/Github/Latent_Style/SchrodingerBridge/_run_t4_queue.ps1"
Start-Process powershell -ArgumentList "-ExecutionPolicy Bypass -File `"$queue`"" -WindowStyle Hidden
Write-Output "LAUNCHED T4 QUEUE"
