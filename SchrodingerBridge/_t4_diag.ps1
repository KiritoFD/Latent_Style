$ErrorActionPreference = 'Continue'
Write-Output '=== PYTHON PROCS ==='
Get-CimInstance Win32_Process -Filter "Name='python.exe'" |
    Select-Object ProcessId, WorkingSetSize, CommandLine |
    Format-List
Write-Output '=== POWERSHELL PROCS ==='
Get-CimInstance Win32_Process -Filter "Name='powershell.exe'" |
    Select-Object ProcessId, CommandLine |
    Format-List
Write-Output '=== EVAL OUTPUT DIR ==='
$d = 'I:/Github/Latent_Style/SchrodingerBridge/exp/p4_fusion_breakout/t4_full_fusion/full_eval_p4_T4_D0_baseline'
if (Test-Path $d) {
    Get-ChildItem -Recurse $d | Select-Object FullName, Length, LastWriteTime | Format-Table -AutoSize
} else {
    Write-Output 'NOT EXISTS'
}
Write-Output '=== RESULT JSON ==='
$r = 'I:/Github/Latent_Style/SchrodingerBridge/exp/p4_fusion_breakout/infer_ablation/T4_D0_baseline.json'
if (Test-Path $r) { Write-Output ('EXISTS size=' + (Get-Item $r).Length) } else { Write-Output 'NOT EXISTS' }
Write-Output '=== LOG SIZE ==='
$l = 'I:/Github/Latent_Style/SchrodingerBridge/exp/p4_fusion_breakout/infer_ablation/T4_D0_baseline.log'
if (Test-Path $l) { Write-Output ('size=' + (Get-Item $l).Length + ' mtime=' + (Get-Item $l).LastWriteTime) } else { Write-Output 'NOT EXISTS' }
