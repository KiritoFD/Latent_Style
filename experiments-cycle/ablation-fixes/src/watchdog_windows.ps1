param(
    [string]$WslDistro = "",
    [string]$ProjectDirWsl = "/mnt/i/Github/Latent_Style/Cycle-NCE/src",
    [string]$InnerScript = "watchdog.sh",
    [string]$ConfigPathWsl = "/mnt/i/Github/Latent_Style/Cycle-NCE/src/config.json",
    [int]$RestartDelaySec = 15,
    [int]$MaxRestarts = 0
)

$ErrorActionPreference = "Continue"

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$winLogDir = Join-Path $scriptDir "..\train_logs_win"
New-Item -ItemType Directory -Force -Path $winLogDir | Out-Null

$stamp = Get-Date -Format "yyyyMMdd_HHmmss"
$logFile = Join-Path $winLogDir ("windows_watchdog_{0}.log" -f $stamp)

function Write-Log {
    param([string]$Message)
    $line = "[{0}] {1}" -f (Get-Date -Format "yyyy-MM-dd HH:mm:ss"), $Message
    $line | Tee-Object -FilePath $logFile -Append
}

function Invoke-WslTrain {
    $cmd = "cd '$ProjectDirWsl' && chmod +x '$InnerScript' && CONFIG_PATH='$ConfigPathWsl' bash './$InnerScript'"
    if ([string]::IsNullOrWhiteSpace($WslDistro)) {
        $output = & wsl.exe bash -lc $cmd 2>&1
    } else {
        $output = & wsl.exe -d $WslDistro bash -lc $cmd 2>&1
    }
    $exitCode = $LASTEXITCODE
    if ($null -ne $output) {
        $output | Tee-Object -FilePath $logFile -Append | Out-Host
    }
    return [int]$exitCode
}

Write-Log "Windows watchdog started. log=$logFile"
$distroLabel = "(default)"
if (-not [string]::IsNullOrWhiteSpace($WslDistro)) {
    $distroLabel = $WslDistro
}
Write-Log "WSL distro=$distroLabel project=$ProjectDirWsl inner=$InnerScript config=$ConfigPathWsl"

$restarts = 0
while ($true) {
    Write-Log "Launching WSL training process..."
    $code = Invoke-WslTrain
    if ($code -eq 0) {
        Write-Log "Training process exited normally (code=0). Stop watchdog."
        break
    }
    $restarts++
    if ($MaxRestarts -gt 0 -and $restarts -gt $MaxRestarts) {
        Write-Log "Reached MaxRestarts=$MaxRestarts. Stop watchdog."
        break
    }
    Write-Log "Training process crashed/exited (code=$code). Restart in $RestartDelaySec sec."
    Start-Sleep -Seconds $RestartDelaySec
}
