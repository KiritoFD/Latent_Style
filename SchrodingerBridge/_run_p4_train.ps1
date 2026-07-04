param(
    [Parameter(Mandatory=$true)][string]$ConfigPath,
    [Parameter(Mandatory=$true)][string]$LogName
)

# Launcher: called via SSH. Launches the wrapper in a hidden window (detached from
# SSH console, survives SSH disconnect) and returns immediately.

$ErrorActionPreference = "Stop"
Set-Location "I:\Github\Latent_Style\SchrodingerBridge"

$logDir = "I:\Github\Latent_Style\SchrodingerBridge\exp\p4_fusion_breakout"
if (-not (Test-Path $logDir)) { New-Item -ItemType Directory -Path $logDir -Force | Out-Null }

$wrapperScript = "I:\Github\Latent_Style\SchrodingerBridge\_run_train_wrapper.ps1"
$pidFile = Join-Path $logDir ($LogName -replace '\.log$', '.pid')
$statusFile = Join-Path $logDir ($LogName -replace '\.log$', '.status')

# Clean old pid/status
foreach ($f in @($pidFile, $statusFile)) {
    if (Test-Path $f) { Remove-Item $f -Force }
}

"LAUNCHING" | Out-File $statusFile -Encoding utf8

# Launch wrapper in a HIDDEN window — this is the key: the hidden window is a new
# process not attached to the SSH console, so it survives SSH disconnect.
$proc = Start-Process -FilePath "powershell.exe" `
    -ArgumentList @('-ExecutionPolicy','Bypass','-NoProfile','-File',$wrapperScript,'-ConfigPath',$ConfigPath,'-LogName',$LogName) `
    -WindowStyle Hidden `
    -PassThru

$proc.Id | Out-File $pidFile -Encoding utf8
"LAUNCHED wrapper_pid=$($proc.Id)" | Out-File $statusFile -Encoding utf8 -Append

Write-Host "Wrapper PID: $($proc.Id)"
Write-Host "PID file: $pidFile"
Write-Host "Training launched in background (hidden window). Monitor status/log files."
