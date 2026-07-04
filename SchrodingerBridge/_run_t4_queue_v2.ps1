# T4 inference ablation queue - detached execution
# Runs 6 remaining experiments sequentially: D3, D2, D5, D6, D1, D0
# Each calls _run_t4_infer.ps1 which handles logging + Tee-Object
# Priority: D3 (train repro) -> D2 (u005) -> D5 (u005_v3) -> D6 (u02_v3) -> D1 (dwt) -> D0 (baseline)

$ErrorActionPreference = "Continue"
Set-Location "I:/Github/Latent_Style/SchrodingerBridge"

$queueLog = "I:/Github/Latent_Style/SchrodingerBridge/exp/p4_fusion_breakout/infer_ablation/_t4_queue_v2.log"
"=== T4 QUEUE START $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') ===" | Out-File $queueLog -Encoding utf8

# Experiment definitions: @(Name, LpMode, Alpha, Kernel, MbMode, Triband, MidScale, HhScale)
$experiments = @(
    @("T4_D3_u01",      "dwt_haar", 0.1,  0,  "single", 0, 0.3, 0.3),
    @("T4_D2_u005",     "dwt_haar", 0.05, 0,  "single", 0, 0.3, 0.3),
    @("T4_D5_u005_v3",  "dwt_haar", 0.05, 16, "single", 0, 0.3, 0.3),
    @("T4_D6_u02_v3",   "dwt_haar", 0.2,  16, "single", 0, 0.3, 0.3),
    @("T4_D1_dwt",      "dwt_haar", 0.0,  0,  "single", 0, 0.3, 0.3),
    @("T4_D0_baseline", "avg_pool", 0.0,  0,  "single", 0, 0.3, 0.3)
)

foreach ($exp in $experiments) {
    $name = $exp[0]
    $resultJson = "I:/Github/Latent_Style/SchrodingerBridge/exp/p4_fusion_breakout/infer_ablation/$name.json"

    # Skip if already done
    if (Test-Path $resultJson) {
        $msg = "[$(Get-Date -Format 'HH:mm:ss')] SKIP $name - result already exists"
        Write-Host $msg
        $msg | Out-File $queueLog -Append -Encoding utf8
        continue
    }

    $msg = "[$(Get-Date -Format 'HH:mm:ss')] START $name params=$($exp -join ',')"
    Write-Host $msg
    $msg | Out-File $queueLog -Append -Encoding utf8

    try {
        & "I:/Github/Latent_Style/SchrodingerBridge/_run_t4_infer.ps1" -ExpName $name -LpMode $exp[1] -Alpha $exp[2] -Kernel $exp[3] -MbMode $exp[4] -Triband $exp[5] -MidScale $exp[6] -HhScale $exp[7]
        $msg = "[$(Get-Date -Format 'HH:mm:ss')] DONE $name exit=$LASTEXITCODE"
    } catch {
        $msg = "[$(Get-Date -Format 'HH:mm:ss')] ERROR $name : $_"
    }
    Write-Host $msg
    $msg | Out-File $queueLog -Append -Encoding utf8

    # Verify result
    if (Test-Path $resultJson) {
        "  -> result JSON written" | Out-File $queueLog -Append -Encoding utf8
    } else {
        "  -> WARNING: result JSON NOT found" | Out-File $queueLog -Append -Encoding utf8
    }
}

"=== T4 QUEUE END $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') ===" | Out-File $queueLog -Append -Encoding utf8
