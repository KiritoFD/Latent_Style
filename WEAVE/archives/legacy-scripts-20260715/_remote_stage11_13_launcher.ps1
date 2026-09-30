# Detached launcher for stage11-13 pipeline
$runner = "C:/Users/Administrator/scripts/_remote_stage11_13_pipeline.ps1"
$proc = Start-Process powershell -ArgumentList @('-ExecutionPolicy','Bypass','-NoProfile','-File',$runner) -WindowStyle Hidden -PassThru
"Launched pipeline PID=$($proc.Id) at $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')"
