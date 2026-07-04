# Run a single T4 experiment directly, capture output to log file
# Usage: _t4_run_one.ps1 <exp_name> <lp_mode> <alpha> <kernel> <mb_mode> <triband> <mid_scale> <hh_scale>
param(
    [Parameter(Mandatory=$true)][string]$ExpName,
    [Parameter(Mandatory=$true)][string]$LpMode,
    [float]$Alpha = 0.0,
    [int]$Kernel = 0,
    [string]$MbMode = "single",
    [int]$Triband = 0,
    [float]$MidScale = 0.3,
    [float]$HhScale = 0.3
)

$env:P4_CKPT_PATH = "I:/Github/Latent_Style/SchrodingerBridge/exp/p4_fusion_breakout/t4_full_fusion/epoch_0001.pt"
$env:P4_CONFIG_PATH = "I:/Github/Latent_Style/SchrodingerBridge/configs/p4_t4_full_fusion.json"
$env:P4_BASELINE_CLIP = "0.7087"
$env:P4_BASELINE_LPIPS = "0.4143"

Set-Location "I:/Github/Latent_Style/SchrodingerBridge"

$logDir = "I:/Github/Latent_Style/SchrodingerBridge/exp/p4_fusion_breakout/infer_ablation"
if (-not (Test-Path $logDir)) { New-Item -ItemType Directory -Path $logDir -Force | Out-Null }
$logFile = Join-Path $logDir "$ExpName.log"

$py = "C:/Program Files/Python312/python.exe"

# Use cmd /c to redirect both stdout and stderr to log file, avoiding PowerShell native command error stream issues
$argStr = "$ExpName $LpMode $Alpha $Kernel $MbMode $Triband $MidScale $HhScale"
Write-Host "Running: $py _p4_infer_ablation.py $argStr"
Write-Host "Log: $logFile"

# Use Start-Process to run python detached, wait for it
$proc = Start-Process -FilePath $py -ArgumentList @("_p4_infer_ablation.py", $ExpName, $LpMode, $Alpha, $Kernel, $MbMode, $Triband, $MidScale, $HhScale) -RedirectStandardOutput $logFile -RedirectStandardError "$logFile.err" -NoNewWindow -PassThru -Wait
Write-Host "Exit code: $($proc.ExitCode)"

$resultJson = Join-Path $logDir "$ExpName.json"
if (Test-Path $resultJson) {
    Write-Host "SUCCESS: $resultJson exists"
    Get-Content $resultJson -Raw
} else {
    Write-Host "FAILED: $resultJson not found"
    Write-Host "=== Last 20 lines of log ==="
    if (Test-Path $logFile) { Get-Content $logFile -Tail 20 }
    Write-Host "=== Last 20 lines of err ==="
    if (Test-Path "$logFile.err") { Get-Content "$logFile.err" -Tail 20 }
}
