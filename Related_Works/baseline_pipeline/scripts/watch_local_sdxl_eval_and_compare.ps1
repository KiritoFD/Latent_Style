param(
    [string]$CurveRoot = "G:\GitHub\Latent_Style\SchrodingerBridge\exp\local_distinct5_512_sdxl_fix_k_b32_e8\full_eval",
    [string]$BaselineSummary = "G:\GitHub\Latent_Style\SchrodingerBridge\exp\distinct5_512_ema_variant_k_content_adaptive_vq_queue_e3_b44_remote\full_eval\epoch_0001\summary.json",
    [string]$CompareScript = "G:\GitHub\Latent_Style\SchrodingerBridge\tools\experiments\compare_distinct5_eval_curve.py",
    [string]$PythonExe = "C:\Users\xy\AppData\Local\Programs\Python\Python312\python.exe",
    [string]$OutputDir = "G:\GitHub\Latent_Style\SchrodingerBridge\docs\experiments\local_distinct5_sdxl_fix_vs_ema_20260605",
    [string]$LogPath = "G:\GitHub\Latent_Style\SchrodingerBridge\_codex_tmp\watch_local_sdxl_eval_and_compare.log",
    [int]$PollSeconds = 30
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

function Write-Log {
    param([string]$Message)
    $timestamp = Get-Date -Format "yyyy-MM-ddTHH:mm:ssK"
    Add-Content -Path $LogPath -Value "[$timestamp] $Message"
}

New-Item -ItemType Directory -Force -Path ([System.IO.Path]::GetDirectoryName($LogPath)) | Out-Null
New-Item -ItemType Directory -Force -Path $OutputDir | Out-Null
Write-Log "watcher start curve_root=$CurveRoot"

$lastCount = -1
while ($true) {
    $summaryFiles = @()
    if (Test-Path -LiteralPath $CurveRoot) {
        $summaryFiles = @(Get-ChildItem -Path $CurveRoot -Filter summary.json -Recurse -ErrorAction SilentlyContinue | Sort-Object FullName)
    }
    $count = $summaryFiles.Count
    if ($count -ne $lastCount -and $count -gt 0) {
        Write-Log "summary count changed to $count; running comparison"
        & $PythonExe $CompareScript `
            --baseline-label "LBM-K e1 (EMA latent)" `
            --baseline-summary $BaselineSummary `
            --curve-label "LBM-K SDXL-fix" `
            --curve-root $CurveRoot `
            --output-dir $OutputDir | Tee-Object -FilePath $LogPath -Append | Out-Null
        $lastCount = $count
    }
    if (Test-Path -LiteralPath (Join-Path $CurveRoot "epoch_0008\summary.json")) {
        Write-Log "detected epoch_0008 summary; watcher complete"
        break
    }
    Write-Log "no final e8 summary yet; sleeping ${PollSeconds}s"
    Start-Sleep -Seconds $PollSeconds
}
