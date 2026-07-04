Set-Location I:\Github\Latent_Style\SchrodingerBridge
$env:PYTHONPATH = ""
$outLog = "I:\Github\Latent_Style\SchrodingerBridge\exp\p4_fusion_breakout\test_train.log"
$errLog = "I:\Github\Latent_Style\SchrodingerBridge\exp\p4_fusion_breakout\test_train_err.log"
$proc = Start-Process -FilePath "C:\Program Files\Python312\python.exe" -ArgumentList "run.py","--config","configs/p4_n11_n16_gate03_whh25.json" -RedirectStandardOutput $outLog -RedirectStandardError $errLog -PassThru
Start-Sleep -Seconds 30
if (-not $proc.HasExited) {
    Write-Output "STILL_RUNNING_KILLING"
    $proc.Kill()
    Start-Sleep -Seconds 2
} else {
    Write-Output "EXITED_CODE=$($proc.ExitCode)"
}
Write-Output "---LOG---"
Get-Content $outLog -ErrorAction SilentlyContinue | Select-Object -First 30
Write-Output "---ERR---"
Get-Content $errLog -ErrorAction SilentlyContinue | Select-Object -First 30
