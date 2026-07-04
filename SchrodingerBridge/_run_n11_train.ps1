Set-Location I:\Github\Latent_Style\SchrodingerBridge
$env:PYTHONPATH = ""
$pythonExe = "C:\Program Files\Python312\python.exe"
$trainArgs = @("run.py", "--config", "configs/p4_n11_n16_gate03_whh25.json")
$outLog = "I:\Github\Latent_Style\SchrodingerBridge\exp\p4_fusion_breakout\n11_n16_train.log"
$errLog = "I:\Github\Latent_Style\SchrodingerBridge\exp\p4_fusion_breakout\n11_n16_train_err.log"
Start-Process -FilePath $pythonExe -ArgumentList $trainArgs -RedirectStandardOutput $outLog -RedirectStandardError $errLog -NoNewWindow
Write-Output "N11_TRAIN_STARTED_DIRECT"
