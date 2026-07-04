# Test Start-Process with python
Set-Location I:/Github/Latent_Style/SchrodingerBridge
$logDir = "I:/Github/Latent_Style/SchrodingerBridge/exp/p4_fusion_breakout/infer_ablation"
$outLog = "$logDir/_startproc_test.log"
$errLog = "$logDir/_startproc_test_err.log"
if (Test-Path $outLog) { Remove-Item $outLog -Force }
if (Test-Path $errLog) { Remove-Item $errLog -Force }
Write-Host "Starting simple python test via Start-Process..."
$pyArgs = @('-c', 'print("hello from python"); import sys; print("python path:", sys.executable); sys.stdout.flush()')
$proc = Start-Process -FilePath 'C:/Program Files/Python312/python.exe' -ArgumentList $pyArgs -RedirectStandardOutput $outLog -RedirectStandardError $errLog -NoNewWindow -PassThru -WorkingDirectory 'I:/Github/Latent_Style/SchrodingerBridge'
Write-Host "PID=$($proc.Id)"
$proc.WaitForExit(15000) | Out-Null
Write-Host "Exited: $($proc.HasExited)"
Start-Sleep -Seconds 2
Write-Host "=== OUT log size ==="
Write-Host (Get-Item $outLog).Length
Write-Host "=== OUT content ==="
if (Test-Path $outLog) { Get-Content $outLog }
Write-Host "=== ERR content ==="
if (Test-Path $errLog) { Get-Content $errLog }
