# Stage10 launcher: uses Start-Process to spawn a detached PowerShell
$logFile = "C:/Users/Administrator/logs/stage10_ll_partial_train.out"
$logDir = Split-Path $logFile -Parent
if (-not (Test-Path $logDir)) { New-Item -ItemType Directory -Path $logDir -Force | Out-Null }

# Write the inner runner script
$runner = @'
Set-Location "I:/Github/Latent_Style/SchrodingerBridge"
$env:PYTHONIOENCODING = "utf-8"
$logFile = "C:/Users/Administrator/logs/stage10_ll_partial_train.out"
"=== STAGE10 LL_PARTIAL TRAIN START $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') ===" | Out-File $logFile -Encoding utf8
& python -u src/run.py --config configs/exp_sty_stage10_ll_partial.json *>&1 | Out-File $logFile -Append -Encoding utf8
"=== STAGE10 LL_PARTIAL TRAIN DONE exit=$LASTEXITCODE $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') ===" | Out-File $logFile -Append -Encoding utf8
'@
$runnerPath = "C:/Users/Administrator/scripts/_stage10_runner.ps1"
$runner | Out-File -FilePath $runnerPath -Encoding utf8

# Spawn detached process
$proc = Start-Process powershell -ArgumentList @('-ExecutionPolicy','Bypass','-NoProfile','-File',$runnerPath) -WindowStyle Hidden -PassThru
"Launched PID=$($proc.Id) at $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')"
