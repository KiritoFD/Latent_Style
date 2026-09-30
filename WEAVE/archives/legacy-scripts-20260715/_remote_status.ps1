$ErrorActionPreference = "SilentlyContinue"
Write-Output "=== PYTHON PROCESSES ==="
Get-Process python | Select-Object Id,StartTime,@{N="WS_MB";E={[math]::Round($_.WorkingSet/1MB,1)}} | Format-Table -AutoSize
Write-Output "=== T11_REPRO LOG TAIL ==="
Get-Content C:\Users\Administrator\logs\t11_repro_train_eval.out -Tail 10
Write-Output "=== T11E2 LOG TAIL ==="
Get-Content C:\Users\Administrator\logs\t11e2_train_eval.out -Tail 10
Write-Output "=== T11_REPRO EVAL DIR ==="
Get-ChildItem I:\Github\Latent_Style\SchrodingerBridge\exp\t11_repro_15ep\full_eval -Recurse -ErrorAction SilentlyContinue | Select-Object FullName,Length | Format-Table -AutoSize
Write-Output "=== T11E2 EVAL DIR ==="
Get-ChildItem I:\Github\Latent_Style\SchrodingerBridge\exp\t11e2_extrap05_15ep\full_eval -Recurse -ErrorAction SilentlyContinue | Select-Object FullName,Length | Format-Table -AutoSize
Write-Output "=== DONE ==="
