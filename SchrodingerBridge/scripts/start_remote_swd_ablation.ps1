param(
    [string]$Root = "I:\Github\Latent_Style\SchrodingerBridge"
)

$ErrorActionPreference = "Stop"
$logDir = Join-Path $Root "logs\remote_swd_ablation"
New-Item -ItemType Directory -Force -Path $logDir | Out-Null

$out = Join-Path $logDir "launcher.out.log"
$err = Join-Path $logDir "launcher.err.log"
$script = Join-Path $Root "scripts\launch_remote_swd_ablation.ps1"
$cmdPath = Join-Path $logDir "launch_remote_swd_ablation.cmd"

$cmdText = @"
@echo off
cd /d "$Root"
powershell -NoProfile -ExecutionPolicy Bypass -File "$script" -Root "$Root" 1> "$out" 2> "$err"
"@
$cmdText | Set-Content -LiteralPath $cmdPath -Encoding ASCII

$taskName = "LatentStyle_SWDAblation"
$startTime = (Get-Date).AddMinutes(1).ToString("HH:mm")
$taskRun = "`"$cmdPath`""

& cmd.exe /c "schtasks /Delete /TN $taskName /F >nul 2>nul"
& schtasks.exe /Create /TN $taskName /SC ONCE /ST $startTime /TR $taskRun /RL HIGHEST /F | Out-Host
& schtasks.exe /Run /TN $taskName | Out-Host

"STARTED task=$taskName cmd=$cmdPath out=$out err=$err"
