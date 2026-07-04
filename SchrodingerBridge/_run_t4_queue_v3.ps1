# T4 inference ablation queue v3 - uses Start-Process -Wait with file redirect
# Avoids PowerShell native command error stream AND SSH pipe bottleneck
# Runs 6 remaining experiments sequentially: D3, D2, D5, D6, D1, D0

$ErrorActionPreference = "Continue"
Set-Location "I:/Github/Latent_Style/SchrodingerBridge"

$env:P4_CKPT_PATH = "I:/Github/Latent_Style/SchrodingerBridge/exp/p4_fusion_breakout/t4_full_fusion/epoch_0001.pt"
$env:P4_CONFIG_PATH = "I:/Github/Latent_Style/SchrodingerBridge/configs/p4_t4_full_fusion.json"
$env:P4_BASELINE_CLIP = "0.7087"
$env:P4_BASELINE_LPIPS = "0.4143"

$logDir = "I:/Github/Latent_Style/SchrodingerBridge/exp/p4_fusion_breakout/infer_ablation"
$queueLog = Join-Path $logDir "_t4_queue_v3.log"
$py = "C:/Program Files/Python312/python.exe"

"=== T4 QUEUE V3 START $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') ===" | Out-File $queueLog -Encoding utf8

# Experiment definitions: @(Name, LpMode, Alpha, Kernel, MbMode, Triband, MidScale, HhScale)
$experiments = @(
    @("T4_D3_u01",      "dwt_haar", "0.1",  "0",  "single", "0", "0.3", "0.3"),
    @("T4_D2_u005",     "dwt_haar", "0.05", "0",  "single", "0", "0.3", "0.3"),
    @("T4_D5_u005_v3",  "dwt_haar", "0.05", "16", "single", "0", "0.3", "0.3"),
    @("T4_D6_u02_v3",   "dwt_haar", "0.2",  "16", "single", "0", "0.3", "0.3"),
    @("T4_D1_dwt",      "dwt_haar", "0.0",  "0",  "single", "0", "0.3", "0.3"),
    @("T4_D0_baseline", "avg_pool", "0.0",  "0",  "single", "0", "0.3", "0.3")
)

foreach ($exp in $experiments) {
    $name = $exp[0]
    $resultJson = Join-Path $logDir "$name.json"

    if (Test-Path $resultJson) {
        "[$(Get-Date -Format 'HH:mm:ss')] SKIP $name - already done" | Out-File $queueLog -Append -Encoding utf8
        continue
    }

    $logFile = Join-Path $logDir "$name.log"
    $errFile = Join-Path $logDir "$name.err"
    $args = @("_p4_infer_ablation.py", $exp[0], $exp[1], $exp[2], $exp[3], $exp[4], $exp[5], $exp[6], $exp[7])

    "[$(Get-Date -Format 'HH:mm:ss')] START $name args=$($exp -join ',')" | Out-File $queueLog -Append -Encoding utf8

    try {
        $proc = Start-Process -FilePath $py -ArgumentList $args -RedirectStandardOutput $logFile -RedirectStandardError $errFile -NoNewWindow -PassThru -Wait
        $exitCode = $proc.ExitCode
        "[$(Get-Date -Format 'HH:mm:ss')] DONE $name exit=$exitCode" | Out-File $queueLog -Append -Encoding utf8
    } catch {
        "[$(Get-Date -Format 'HH:mm:ss')] ERROR $name : $_" | Out-File $queueLog -Append -Encoding utf8
    }

    if (Test-Path $resultJson) {
        "  -> SUCCESS: $name.json written" | Out-File $queueLog -Append -Encoding utf8
    } else {
        "  -> FAILED: $name.json NOT found" | Out-File $queueLog -Append -Encoding utf8
    }
}

"=== T4 QUEUE V3 END $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') ===" | Out-File $queueLog -Append -Encoding utf8
