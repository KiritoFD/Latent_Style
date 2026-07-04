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

# T4 checkpoint (corrected: no checkpoints/ subdir)
$env:P4_CKPT_PATH = "I:/Github/Latent_Style/SchrodingerBridge/exp/p4_fusion_breakout/t4_full_fusion/epoch_0001.pt"
$env:P4_CONFIG_PATH = "I:/Github/Latent_Style/SchrodingerBridge/configs/p4_t4_full_fusion.json"
$env:P4_BASELINE_CLIP = "0.7087"
$env:P4_BASELINE_LPIPS = "0.4143"

Set-Location "I:/Github/Latent_Style/SchrodingerBridge"

$logDir = "I:/Github/Latent_Style/SchrodingerBridge/exp/p4_fusion_breakout/infer_ablation"
if (-not (Test-Path $logDir)) { New-Item -ItemType Directory -Path $logDir -Force | Out-Null }
$logFile = Join-Path $logDir "$ExpName.log"

Write-Host "=== T4 Inference Ablation ==="
Write-Host "ExpName : $ExpName"
Write-Host "LpMode  : $LpMode"
Write-Host "Alpha   : $Alpha"
Write-Host "Kernel  : $Kernel"
Write-Host "MbMode  : $MbMode"
Write-Host "Triband : $Triband"
Write-Host "MidScale: $MidScale"
Write-Host "HhScale : $HhScale"
Write-Host "Ckpt    : $env:P4_CKPT_PATH"
Write-Host "Config  : $env:P4_CONFIG_PATH"
Write-Host "Log     : $logFile"
Write-Host "Start   : $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')"

$py = "C:/Program Files/Python312/python.exe"
& $py _p4_infer_ablation.py $ExpName $LpMode $Alpha $Kernel $MbMode $Triband $MidScale $HhScale 2>&1 | Tee-Object -FilePath $logFile

Write-Host "End     : $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')"
Write-Host "=== Done: $ExpName ==="
