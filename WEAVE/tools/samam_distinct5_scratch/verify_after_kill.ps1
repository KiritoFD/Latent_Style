$ErrorActionPreference = 'SilentlyContinue'

Write-Host "===== GPU status ====="
nvidia-smi

Write-Host ""
Write-Host "===== ALL python processes ====="
Get-CimInstance Win32_Process -Filter "Name = 'python.exe' OR Name = 'pythonw.exe'" |
    Select-Object ProcessId, ParentProcessId, CreationDate, CommandLine | Format-List

Write-Host ""
Write-Host "===== ALL cmd.exe with 628/wsl/persistent ====="
Get-CimInstance Win32_Process -Filter "Name = 'cmd.exe'" |
    Where-Object { $_.CommandLine -match '_628_|628_run|destructive|clean_base|p7_runner|p8d|p8e|p8f|wsl_persistent_holder|test_wsl' } |
    Select-Object ProcessId, ParentProcessId, CommandLine | Format-List

Write-Host ""
Write-Host "===== ALL powershell processes ====="
Get-CimInstance Win32_Process -Filter "Name = 'powershell.exe' OR Name = 'pwsh.exe'" |
    Select-Object ProcessId, ParentProcessId, CreationDate, CommandLine | Format-List

Write-Host ""
Write-Host "===== ALL scheduled tasks status (628/samam/destructive) ====="
$checkTasks = @('destructive_ablation','p8d_color_deep','p8e_clean_validate','sb_628_batch_runner','sb_628_eval_all_runner','sb_p7_runner','SaMamFaithfulP8Remote','SaMAM_Distinct5_B6_Long_WSL','SaMAM_Distinct5_B6_Long_WSL_Detached','SaMAM_Distinct5_B6_Seg250_WSL','SaMAM_Distinct5_B8_Seg250_WSL')
foreach ($tn in $checkTasks) {
    $out = & schtasks /Query /TN $tn /FO LIST 2>&1 | Out-String
    Write-Host "--- $tn ---"
    Write-Host $out
}

Write-Host ""
Write-Host "===== Last 20 lines of watchdog.log ====="
$wdlog = "I:\Github\Latent_Style\SchrodingerBridge\exp\628_ablation\destructive_logs\watchdog.log"
if (Test-Path $wdlog) { Get-Content $wdlog -Tail 20 } else { Write-Host "(no watchdog.log)" }

Write-Host ""
Write-Host "===== Last 20 lines of p7_runner.log ====="
$p7log = "I:\Github\Latent_Style\SchrodingerBridge\exp\628_ablation\destructive_logs\p7_runner.log"
if (Test-Path $p7log) { Get-Content $p7log -Tail 20 } else { Write-Host "(no p7_runner.log)" }

Write-Host ""
Write-Host "===== Last 20 lines of p8d_launcher.log ====="
$p8dlog = "I:\Github\Latent_Style\SchrodingerBridge\exp\628_ablation\destructive_logs\p8d_launcher.log"
if (Test-Path $p8dlog) { Get-Content $p8dlog -Tail 20 } else { Write-Host "(no p8d_launcher.log)" }

Write-Host ""
Write-Host "===== Last 20 lines of batch_log.txt ====="
$bl = "I:\Github\Latent_Style\SchrodingerBridge\exp\628_ablation\destructive_logs\batch_log.txt"
if (Test-Path $bl) { Get-Content $bl -Tail 20 } else { Write-Host "(no batch_log.txt)" }

Write-Host ""
Write-Host "===== clean_base validate.log tail ====="
$vlog = "I:\Github\Latent_Style\SchrodingerBridge\exp\clean_base\validate.log"
if (Test-Path $vlog) { Get-Content $vlog -Tail 15 } else { Write-Host "(no validate.log)" }
