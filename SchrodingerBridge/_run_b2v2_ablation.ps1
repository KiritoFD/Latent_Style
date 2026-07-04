param(
    [Parameter(Mandatory=$true)][string]$ExpName,
    [string]$LpMode = 'avg_pool',
    [float]$Alpha = 0.0,
    [int]$Kernel = 0,
    [string]$MbMode = 'single',
    [int]$Triband = 0,
    [float]$MidScale = 0.3,
    [float]$HhScale = 0.3
)

# FC-SB Phase 4: B2 V2 ablation runner.
# Sets env vars to switch _p4_infer_ablation.py onto B2 V2 checkpoint.

$env:P4_CKPT_PATH = "I:/Github/Latent_Style/SchrodingerBridge/exp/620_spectral_v2_weights/epoch_0001.pt"
$env:P4_CONFIG_PATH = "I:/Github/Latent_Style/SchrodingerBridge/configs/620_spectral_v2_weights.json"
$env:P4_BASELINE_CLIP = "0.6731"
$env:P4_BASELINE_LPIPS = "0.2781"

Set-Location I:/Github/Latent_Style/SchrodingerBridge

$outDir = "I:/Github/Latent_Style/SchrodingerBridge/exp/p4_fusion_breakout/infer_ablation"
if (-not (Test-Path $outDir)) { New-Item -ItemType Directory -Path $outDir -Force | Out-Null }
$logPath = Join-Path $outDir "$ExpName.log"

Write-Host "[run_b2v2] ExpName=$ExpName LpMode=$LpMode Alpha=$Alpha Kernel=$Kernel MbMode=$MbMode Triband=$Triband MidScale=$MidScale HhScale=$HhScale"
Write-Host "[run_b2v2] P4_CKPT_PATH=$env:P4_CKPT_PATH"
Write-Host "[run_b2v2] P4_CONFIG_PATH=$env:P4_CONFIG_PATH"
Write-Host "[run_b2v2] log=$logPath"

& "C:/Program Files/Python312/python.exe" _p4_infer_ablation.py $ExpName $LpMode $Alpha $Kernel $MbMode $Triband $MidScale $HhScale 2>&1 | Tee-Object -FilePath $logPath

Write-Host "[run_b2v2] DONE exit=$LASTEXITCODE"
