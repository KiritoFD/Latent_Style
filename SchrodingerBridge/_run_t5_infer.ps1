param(
    [Parameter(Mandatory=$true, Position=0)][string]$ExpName,
    [Parameter(Position=1)][string]$LpMode = "avg_pool",
    [Parameter(Position=2)][float]$Alpha = 0.0,
    [Parameter(Position=3)][int]$Kernel = 0,
    [Parameter(Position=4)][string]$MbMode = "single",
    [Parameter(Position=5)][int]$Triband = 0,
    [Parameter(Position=6)][float]$MidScale = 0.3,
    [Parameter(Position=7)][float]$HhScale = 0.3
)

# T5 checkpoint (path verified: NO checkpoints/ subdir, .pt files are directly under t5_b2v2_d2_d4/)
$env:P4_CKPT_PATH = "I:/Github/Latent_Style/SchrodingerBridge/exp/p4_fusion_breakout/t5_b2v2_d2_d4/epoch_0007.pt"
$env:P4_CONFIG_PATH = "I:/Github/Latent_Style/SchrodingerBridge/configs/p4_t5_b2v2_d2_d4.json"
$env:P4_BASELINE_CLIP = "0.7016"
$env:P4_BASELINE_LPIPS = "0.3515"

Set-Location "I:/Github/Latent_Style/SchrodingerBridge"

$logDir = "exp/p4_fusion_breakout/infer_ablation"
if (-not (Test-Path $logDir)) { New-Item -ItemType Directory -Force -Path $logDir | Out-Null }
$logPath = "$logDir/$ExpName.log"

Write-Host "[t5_infer] exp=$ExpName lp=$LpMode alpha=$Alpha kernel=$Kernel mb=$MbMode triband=$Triband mid=$MidScale hh=$HhScale"
Write-Host "[t5_infer] ckpt=$env:P4_CKPT_PATH"
Write-Host "[t5_infer] config=$env:P4_CONFIG_PATH"
Write-Host "[t5_infer] start=$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')"

& "C:/Program Files/Python312/python.exe" _p4_infer_ablation.py $ExpName $LpMode $Alpha $Kernel $MbMode $Triband $MidScale $HhScale 2>&1 | Tee-Object -FilePath $logPath

Write-Host "[t5_infer] end=$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')"
Write-Host "[t5_infer] DONE_MARKER=$ExpName"
