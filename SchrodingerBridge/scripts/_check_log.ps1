# Check pipeline log for eval results
$log = "I:\Github\Latent_Style\SchrodingerBridge\logs\pipeline_fill_main.log"
Write-Host "=== Last 60 lines of pipeline log ==="
Get-Content $log -Tail 60
Write-Host ""
Write-Host "=== Lines with eval/CLIP/SKIP/FAIL ==="
Get-Content $log | Where-Object { $_ -match 'eval-|CLIP-S|SKIP|FAILED|Phase C|Phase D|COMPLETED' }
