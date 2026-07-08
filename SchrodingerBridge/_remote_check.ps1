$ErrorActionPreference = 'SilentlyContinue'
Write-Output "=== GPU ==="
nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu --format=csv,noheader
Write-Output "=== PYTHON PROCS ==="
Get-CimInstance Win32_Process -Filter "Name='python.exe'" |
    Select-Object ProcessId, @{N='MemMB';E={[math]::Round($_.WorkingSetSize/1MB,0)}}, CommandLine |
    Format-List
Write-Output "=== RUNNER LOG TAIL ==="
if (Test-Path "I:\Github\Latent_Style\SchrodingerBridge\remote_ablation_log.txt") {
    Get-Content "I:\Github\Latent_Style\SchrodingerBridge\remote_ablation_log.txt" -Tail 30
}
Write-Output "=== PID FILE ==="
$pidf = "I:\Github\Latent_Style\SchrodingerBridge\remote_ablation_runner.pid"
if (Test-Path $pidf) { Get-Content $pidf } else { Write-Output "(no pid file)" }
