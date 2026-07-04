param(
    [Parameter(Mandatory=$true, Position=0)][string]$ExpName,
    [Parameter(Position=1)][string]$Mode='avg_pool',
    [Parameter(Position=2)][string]$Alpha='0',
    [Parameter(Position=3)][string]$Kernel='0',
    [Parameter(Position=4)][string]$MbMode='single',
    [Parameter(Position=5)][string]$Triband='0',
    [Parameter(Position=6)][string]$MidScale='0.3',
    [Parameter(Position=7)][string]$HhScale='0.3'
)
# FC-SB Phase 4 B2 V2 inference ablation launcher
# Sets B2 V2 env vars and launches _p4_infer_ablation.py in the background.
$env:P4_CKPT_PATH = "I:/Github/Latent_Style/SchrodingerBridge/exp/620_spectral_v2_weights/epoch_0001.pt"
$env:P4_CONFIG_PATH = "I:/Github/Latent_Style/SchrodingerBridge/configs/620_spectral_v2_weights.json"
$env:P4_BASELINE_CLIP = "0.6731"
$env:P4_BASELINE_LPIPS = "0.2781"
Set-Location I:/Github/Latent_Style/SchrodingerBridge
$logDir = "I:/Github/Latent_Style/SchrodingerBridge/exp/p4_fusion_breakout/infer_ablation"
New-Item -ItemType Directory -Force -Path $logDir | Out-Null
$outLog = "$logDir/${ExpName}.log"
$errLog = "$logDir/${ExpName}_err.log"
# Remove old result/log files so polling can detect completion cleanly
$resultJson = "$logDir/${ExpName}.json"
if (Test-Path $resultJson) { Remove-Item $resultJson -Force }
if (Test-Path $outLog) { Remove-Item $outLog -Force }
if (Test-Path $errLog) { Remove-Item $errLog -Force }
$pyArgs = @('_p4_infer_ablation.py', $ExpName, $Mode, $Alpha, $Kernel, $MbMode, $Triband, $MidScale, $HhScale)
$proc = Start-Process -FilePath 'C:/Program Files/Python312/python.exe' -ArgumentList $pyArgs -RedirectStandardOutput $outLog -RedirectStandardError $errLog -NoNewWindow -PassThru
Write-Output "PID=$($proc.Id) EXP=$ExpName OUT=$outLog ERR=$errLog"
