$logFile = "C:\Users\Administrator\logs\refactor_verify_eval.log"
Get-Content $logFile | Select-String -Pattern "strict|missing|unexpected|asg|skip|load_state" | ForEach-Object { $_.Line }
