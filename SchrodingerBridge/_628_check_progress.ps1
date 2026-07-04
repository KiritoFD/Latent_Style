# Check remote 628 batch progress
$sshCmd = 'type I:\Github\Latent_Style\SchrodingerBridge\exp\628_ablation\destructive_logs\batch_runner_stdout.log'
$output = ssh -o ConnectTimeout=15 Administrator@100.115.18.62 -p 2222 $sshCmd
if ($LASTEXITCODE -ne 0) {
    Write-Host "SSH failed (exit=$LASTEXITCODE)"
    exit 1
}
$lines = $output -split "`r?`n"
$done = ($lines | Select-String -Pattern '\[.+\] DONE').Count
$start = ($lines | Select-String -Pattern '\[.+\] START').Count
$fail = ($lines | Select-String -Pattern '\[.+\] FAIL').Count
Write-Host "=== Batch Summary ==="
Write-Host "DONE: $done | START: $start | FAIL: $fail"
Write-Host ""
Write-Host "=== Last 40 lines (DONE/START/FAIL only) ==="
$lines | Select-String -Pattern '\[.+\] (DONE|START|FAIL)' | Select-Object -Last 40 | ForEach-Object { Write-Host $_.Line }
