$evalDir = "I:\Github\Latent_Style\SchrodingerBridge\exp\clean_base_v2\full_eval\epoch_0010"
Write-Output "=== All files in eval dir ==="
Get-ChildItem $evalDir -Recurse -File | Format-Table Name, Length, LastWriteTime
Write-Output ""
Write-Output "=== Looking for summary files ==="
Get-ChildItem $evalDir -Recurse -File -Filter "*summary*" | ForEach-Object { Write-Output $_.FullName; Get-Content $_.FullName }
Write-Output ""
Write-Output "=== Looking for json files ==="
Get-ChildItem $evalDir -Recurse -File -Filter "*.json" | ForEach-Object { Write-Output "--- $($_.Name) ---"; Get-Content $_.FullName -TotalCount 50 }
Write-Output ""
Write-Output "=== LOG (last 60 lines) ==="
$logPath = "I:\Github\Latent_Style\SchrodingerBridge\eval_stage1_log.txt"
if (Test-Path $logPath) {
    Get-Content $logPath -Tail 60
}
