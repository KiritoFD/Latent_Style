param(
    [Parameter(Mandatory=$true, Position=0)][string]$ExpName
)
# FC-SB Phase 4 B2 V2 foreground worker - runs ONE experiment to completion.
# Designed to be called via SSH (one SSH session per experiment for parallelism).
$env:P4_CKPT_PATH = "I:/Github/Latent_Style/SchrodingerBridge/exp/620_spectral_v2_weights/epoch_0001.pt"
$env:P4_CONFIG_PATH = "I:/Github/Latent_Style/SchrodingerBridge/configs/620_spectral_v2_weights.json"
$env:P4_BASELINE_CLIP = "0.6731"
$env:P4_BASELINE_LPIPS = "0.2781"
Set-Location I:/Github/Latent_Style/SchrodingerBridge

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
$pyArgs = @('_p4_infer_ablation.py', $ExpName) + $p
Write-Host "=== FG WORKER: $ExpName ==="
Write-Host "Args: $pyArgs"
Write-Host "CWD: $(Get-Location)"
& "C:/Program Files/Python312/python.exe" @pyArgs
Write-Host "=== FG WORKER DONE: $ExpName exit=$LASTEXITCODE ==="
