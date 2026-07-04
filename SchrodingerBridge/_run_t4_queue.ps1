$ErrorActionPreference = 'Continue'
$env:P4_CKPT_PATH = "I:/Github/Latent_Style/SchrodingerBridge/exp/p4_fusion_breakout/t4_full_fusion/epoch_0001.pt"
$env:P4_CONFIG_PATH = "I:/Github/Latent_Style/SchrodingerBridge/configs/p4_t4_full_fusion.json"
$env:P4_BASELINE_CLIP = "0.7087"
$env:P4_BASELINE_LPIPS = "0.4143"

Set-Location "I:/Github/Latent_Style/SchrodingerBridge"
$logDir = "I:/Github/Latent_Style/SchrodingerBridge/exp/p4_fusion_breakout/infer_ablation"
$py = "C:\Program Files\Python312\python.exe"
$queueLog = Join-Path $logDir "_t4_queue.log"

function Run-One($name, $lp, $al, $k, $mb, $tb, $mid, $hh) {
    $logFile = Join-Path $logDir "$name.log"
    $doneFile = Join-Path $logDir "$name.done"
    $resultJson = Join-Path $logDir "$name.json"
    if (Test-Path $resultJson) { Add-Content $queueLog "SKIP $name (result exists)"; return }
    Remove-Item $doneFile -ErrorAction SilentlyContinue
    $msg = "=== START $name lp=$lp al=$al k=$k mb=$mb tb=$tb mid=$mid hh=$hh : $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')"
    Add-Content $queueLog $msg
    # cmd /c with FD-level redirection avoids PowerShell native-command stderr mangling
    $argStr = "_p4_infer_ablation.py $name $lp $al $k $mb $tb $mid $hh"
    $redirCmd = "`"$py`" $argStr > `"$logFile`" 2>&1"
    cmd /c $redirCmd
    $ec = $LASTEXITCODE
    "$ec" | Out-File $doneFile -Encoding ASCII
    Add-Content $queueLog "=== DONE $name exit=$ec : $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')"
}

# Priority order: most likely breakthrough first
Run-One 'T4_D4_u01_v3'   'dwt_haar' 0.1  16 'single' 0 0.3 0.3   # U+V, training-matched alpha
Run-One 'T4_D3_u01'      'dwt_haar' 0.1  0  'single' 0 0.3 0.3   # training match (true baseline)
Run-One 'T4_D2_u005'     'dwt_haar' 0.05 0  'single' 0 0.3 0.3   # lighter U
Run-One 'T4_D5_u005_v3'  'dwt_haar' 0.05 16 'single' 0 0.3 0.3   # lighter U+V
Run-One 'T4_D6_u02_v3'   'dwt_haar' 0.2  16 'single' 0 0.3 0.3   # aggressive U+V
Run-One 'T4_D1_dwt'      'dwt_haar' 0    0  'single' 0 0.3 0.3   # DWT only, no extrap
Run-One 'T4_D0_baseline' 'avg_pool' 0    0  'single' 0 0.3 0.3   # no-infer-params baseline

Add-Content $queueLog "=== ALL T4 QUEUE DONE : $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')"
