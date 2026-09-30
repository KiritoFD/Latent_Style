param(
    [string]$TrainPidFile = "G:\GitHub\Latent_Style\SchrodingerBridge\_codex_tmp\local_distinct5_sdxl_fix_train.pid",
    [string]$LogPath = "G:\GitHub\Latent_Style\SchrodingerBridge\_codex_tmp\watch_local_sdxl_then_launch_img2img_turbo_smoke.log",
    [string]$PythonExe = "C:\Users\xy\AppData\Local\Programs\Python\Python312\python.exe",
    [string]$LauncherScript = "G:\GitHub\Latent_Style\Related_Works\baseline_pipeline\scripts\run_img2img_turbo_distinct5_smoke.py",
    [string]$Target = "Early_Renaissance",
    [string]$DatasetsRoot = "F:\wikiart_distinct5_img2img_turbo_datasets",
    [string]$RunRoot = "G:\GitHub\Latent_Style\Related_Works\runs\img2img_turbo_distinct5_smoke_auto",
    [int]$GpuMemoryQuietMiB = 1500,
    [int]$GpuUtilQuietPercent = 15,
    [int]$PollSeconds = 30
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

function Write-Log {
    param([string]$Message)
    $timestamp = Get-Date -Format "yyyy-MM-ddTHH:mm:ssK"
    Add-Content -Path $LogPath -Value "[$timestamp] $Message"
}

function Test-ProcessAlive {
    param([int]$ProcessId)
    try {
        $null = Get-Process -Id $ProcessId -ErrorAction Stop
        return $true
    }
    catch {
        return $false
    }
}

function Get-GpuState {
    $line = & nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits 2>$null | Select-Object -First 1
    if (-not $line) {
        return $null
    }
    $parts = $line -split ","
    if ($parts.Count -lt 2) {
        return $null
    }
    return [pscustomobject]@{
        MemoryUsedMiB = [int]($parts[0].Trim())
        UtilPercent   = [int]($parts[1].Trim())
    }
}

New-Item -ItemType Directory -Force -Path ([System.IO.Path]::GetDirectoryName($LogPath)) | Out-Null
Write-Log "watcher start target=$Target train_pid_file=$TrainPidFile"

$trainPid = $null
if (Test-Path -LiteralPath $TrainPidFile) {
    try {
        $trainPid = [int](Get-Content -LiteralPath $TrainPidFile | Select-Object -First 1)
        Write-Log "observed train pid $trainPid"
    }
    catch {
        Write-Log "failed to parse train pid file; will fall back to GPU quiet wait"
    }
}
else {
    Write-Log "train pid file missing; will fall back to GPU quiet wait"
}

if ($trainPid) {
    while (Test-ProcessAlive -ProcessId $trainPid) {
        Write-Log "train pid $trainPid still alive; sleeping ${PollSeconds}s"
        Start-Sleep -Seconds $PollSeconds
    }
    Write-Log "train pid $trainPid exited"
}

while ($true) {
    $gpuState = Get-GpuState
    if ($null -eq $gpuState) {
        Write-Log "gpu state unavailable; sleeping ${PollSeconds}s"
        Start-Sleep -Seconds $PollSeconds
        continue
    }
    if ($gpuState.MemoryUsedMiB -le $GpuMemoryQuietMiB -and $gpuState.UtilPercent -le $GpuUtilQuietPercent) {
        Write-Log "gpu quiet enough memory=${($gpuState.MemoryUsedMiB)}MiB util=${($gpuState.UtilPercent)}%"
        break
    }
    Write-Log "gpu still busy memory=${($gpuState.MemoryUsedMiB)}MiB util=${($gpuState.UtilPercent)}%; sleeping ${PollSeconds}s"
    Start-Sleep -Seconds $PollSeconds
}

$arguments = @(
    $LauncherScript,
    "--target", $Target,
    "--datasets-root", $DatasetsRoot,
    "--run-root", $RunRoot,
    "--run"
)
Write-Log ("launching smoke: " + ($arguments -join " "))
$proc = Start-Process -FilePath $PythonExe -ArgumentList $arguments -WindowStyle Hidden -Wait -PassThru
Write-Log "smoke exit code $($proc.ExitCode)"
