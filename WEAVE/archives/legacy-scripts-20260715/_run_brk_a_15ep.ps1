# Launch training via schtasks (reliable background execution on Windows)
$taskName = "brk_a_15ep"
$workDir = "C:\Users\Administrator\SchrodingerBridge\src"
$cmd = "powershell -NoProfile -Command `"Set-Location '$workDir'; python run.py *>> C:\Users\Administrator\logs\brk_a_15ep_train.log`""

# Remove existing task if any
schtasks /delete /tn $taskName /f 2>$null

# Create and run task
schtasks /create /tn $taskName /tr $cmd /sc once /st 00:00 /rl highest /f
schtasks /run /tn $taskName
Write-Output "TASK_STARTED"
