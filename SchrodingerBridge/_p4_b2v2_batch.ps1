param(
    [Parameter(Mandatory=$true, Position=0)][string]$ExpName
)
# FC-SB Phase 4 B2 V2 inference ablation batch launcher.
# Looks up experiment params by name, sets B2 V2 env vars, runs in background.
$env:P4_CKPT_PATH = "I:/Github/Latent_Style/SchrodingerBridge/exp/620_spectral_v2_weights/epoch_0001.pt"
$env:P4_CONFIG_PATH = "I:/Github/Latent_Style/SchrodingerBridge/configs/620_spectral_v2_weights.json"
$env:P4_BASELINE_CLIP = "0.6731"
$env:P4_BASELINE_LPIPS = "0.2781"
Set-Location I:/Github/Latent_Style/SchrodingerBridge

# Experiment parameter lookup table
# Format: exp_name -> @(lowpass_mode, style_extrap_alpha, patch_adain_kernel, multiband_adain_mode, tri_band_lock, mid_scale, hh_scale)
$expTable = @{
    "D10_b2_baseline"      = @("avg_pool",  "0",   "0",  "single", "0", "0.3", "0.3")
    "D11_b2_u4"            = @("avg_pool",  "0.1", "0",  "single", "0", "0.3", "0.3")
    "D12_b2_v3"            = @("avg_pool",  "0",   "16", "single", "0", "0.3", "0.3")
    "D13_b2_u4_v3"         = @("avg_pool",  "0.1", "16", "single", "0", "0.3", "0.3")
    "D14_b2_u4_v3_dwt"     = @("dwt_haar",  "0.1", "16", "single", "0", "0.3", "0.3")
    "D15_b2_u4_v3_dwt_a02" = @("dwt_haar",  "0.2", "16", "single", "0", "0.3", "0.3")
}

if (-not $expTable.ContainsKey($ExpName)) {
    Write-Error "Unknown experiment: $ExpName"
    exit 2
}

$p = $expTable[$ExpName]
$logDir = "I:/Github/Latent_Style/SchrodingerBridge/exp/p4_fusion_breakout/infer_ablation"
New-Item -ItemType Directory -Force -Path $logDir | Out-Null
$outLog = "$logDir/${ExpName}.log"
$errLog = "$logDir/${ExpName}_err.log"
$resultJson = "$logDir/${ExpName}.json"
# Clean old artifacts so polling detects fresh completion
if (Test-Path $resultJson) { Remove-Item $resultJson -Force }
if (Test-Path $outLog) { Remove-Item $outLog -Force }
if (Test-Path $errLog) { Remove-Item $errLog -Force }

$pyArgs = @('_p4_infer_ablation.py', $ExpName) + $p
Write-Host "Launching: python $pyArgs"
$proc = Start-Process -FilePath 'C:/Program Files/Python312/python.exe' -ArgumentList $pyArgs -RedirectStandardOutput $outLog -RedirectStandardError $errLog -NoNewWindow -PassThru
Write-Output "PID=$($proc.Id) EXP=$ExpName OUT=$outLog ERR=$errLog"
