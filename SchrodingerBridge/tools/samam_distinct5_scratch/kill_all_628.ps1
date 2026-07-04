$ErrorActionPreference = 'SilentlyContinue'

Write-Host "===== STEP 1: Disable all 628/samam scheduled tasks ====="
$taskNames = @(
    'destructive_ablation',
    'p8d_color_deep',
    'p8e_clean_validate',
    'sb_628_batch_runner',
    'sb_628_eval_all_runner',
    'sb_p7_runner',
    'SaMAM_Distinct5_B6_Long_WSL',
    'SaMAM_Distinct5_B6_Long_WSL_Detached',
    'SaMAM_Distinct5_B6_Seg250_WSL',
    'SaMAM_Distinct5_B8_Seg250_WSL',
    'SaMamFaithfulP8Remote',
    'latent-samam-distinct5-convergence-20260606_222608',
    'latent-samam-distinct5-convergence-20260607_002420',
    'latent-samam-distinct5-convergence-20260607_011328',
    'latent-samam-distinct5-same-cost-20260606_034359',
    'latent-samam-distinct5-same-cost-20260606_133519',
    'latent-samam-distinct5-same-cost-20260606_133730',
    'latent-samam-distinct5-same-cost-20260606_155105',
    'latent-samam-distinct5-same-cost-20260606_162544',
    'latent-samam-distinct5-same-cost-20260606_162933',
    'samam-latent-step0020-fast-patch1-20260606-1',
    'samam-latent-step0050-fast-20260606-1',
    'samam-latent-step0050-fast-patch4-20260606-1',
    'samam-latent-step0110-fast-patch1-20260606-1',
    'samam-latent-step0150-fast-20260606-1',
    'samam-latent-step0150-fast-patch4-20260606-1',
    'samam-step0050-fast-eval-20260606'
)
foreach ($tn in $taskNames) {
    Write-Host "Disabling: $tn"
    & schtasks /Change /TN $tn /DISABLE 2>&1 | Out-String | Write-Host
}

Write-Host ""
Write-Host "===== STEP 2: Kill all 628-related python processes ====="
$pyToKill = Get-CimInstance Win32_Process -Filter "Name = 'python.exe' OR Name = 'pythonw.exe'" |
    Where-Object { $_.CommandLine -match '628|destructive|clean_base|SchrodingerBridge.*run\.py|run_evaluation' }
foreach ($p in $pyToKill) {
    Write-Host "Killing python PID=$($p.ProcessId): $($p.CommandLine.Substring(0, [Math]::Min(120, $_.CommandLine.Length)))"
    Stop-Process -Id $p.ProcessId -Force -ErrorAction SilentlyContinue
}

Write-Host ""
Write-Host "===== STEP 3: Kill all 628-related cmd.exe (launcher bats) ====="
$cmdToKill = Get-CimInstance Win32_Process -Filter "Name = 'cmd.exe'" |
    Where-Object { $_.CommandLine -match '_628_|628_run|destructive|clean_base|p7_runner|p8d|p8e|p8f' }
foreach ($c in $cmdToKill) {
    Write-Host "Killing cmd PID=$($c.ProcessId): $($c.CommandLine)"
    Stop-Process -Id $c.ProcessId -Force -ErrorAction SilentlyContinue
}

Write-Host ""
Write-Host "===== STEP 4: Kill wsl_persistent_holder cmd loops (they are crash-looping) ====="
$wslHolder = Get-CimInstance Win32_Process -Filter "Name = 'cmd.exe'" |
    Where-Object { $_.CommandLine -match 'wsl_persistent_holder' }
foreach ($w in $wslHolder) {
    Write-Host "Killing wsl_persistent_holder cmd PID=$($w.ProcessId)"
    Stop-Process -Id $w.ProcessId -Force -ErrorAction SilentlyContinue
}

Write-Host ""
Write-Host "===== STEP 5: Kill test_wsl_cmd.ps1 powershell (PID 16784) and its cmd parent (7464) ====="
Stop-Process -Id 16784 -Force -ErrorAction SilentlyContinue
Stop-Process -Id 7464 -Force -ErrorAction SilentlyContinue
Write-Host "Killed 16784 (powershell test_wsl_cmd) and 7464 (its cmd parent)"

Write-Host ""
Write-Host "===== STEP 6: Kill orphaned nvidia-smi monitor loops ====="
$nvidiaLoops = Get-CimInstance Win32_Process -Filter "Name = 'cmd.exe'" |
    Where-Object { $_.CommandLine -match 'nvidia-smi.*-l' }
foreach ($n in $nvidiaLoops) {
    Write-Host "Killing nvidia-smi loop cmd PID=$($n.ProcessId)"
    Stop-Process -Id $n.ProcessId -Force -ErrorAction SilentlyContinue
}

Write-Host ""
Write-Host "===== STEP 7: Wait 5s and verify ====="
Start-Sleep 5

Write-Host ""
Write-Host "===== Remaining python processes ====="
$remain = Get-CimInstance Win32_Process -Filter "Name = 'python.exe' OR Name = 'pythonw.exe'"
if ($remain) {
    $remain | Select-Object ProcessId, ParentProcessId, CommandLine | Format-List
} else {
    Write-Host "NONE - all python processes killed"
}

Write-Host ""
Write-Host "===== Remaining 628-related cmd.exe ====="
$remainCmd = Get-CimInstance Win32_Process -Filter "Name = 'cmd.exe'" |
    Where-Object { $_.CommandLine -match '_628_|628_run|destructive|clean_base|p7_runner|p8d|p8e|p8f|wsl_persistent_holder' }
if ($remainCmd) {
    $remainCmd | Select-Object ProcessId, CommandLine | Format-List
} else {
    Write-Host "NONE - all 628/wsl_holder cmd killed"
}

Write-Host ""
Write-Host "===== GPU status after cleanup ====="
nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv

Write-Host ""
Write-Host "===== Verify scheduled tasks disabled ====="
foreach ($tn in @('destructive_ablation','p8d_color_deep','p8e_clean_validate','sb_628_batch_runner','sb_628_eval_all_runner','sb_p7_runner')) {
    $status = & schtasks /Query /TN $tn /FO LIST 2>$null | Select-String 'Status:'
    Write-Host "$tn : $status"
}
