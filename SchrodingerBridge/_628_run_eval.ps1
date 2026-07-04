# Sync eval script, pause batch, run eval, resume batch
$ErrorActionPreference = 'Continue'
$remote = 'Administrator@100.115.18.62'
$port = '2222'
$remoteRoot = 'I:\Github\Latent_Style\SchrodingerBridge'
$localRoot = 'g:\GitHub\Latent_Style\SchrodingerBridge'

function Remote-Cmd($cmd) {
    ssh -o ConnectTimeout=15 $remote -p $port $cmd
}

# Step 1: Sync eval script
Write-Host "=== Step 1: Syncing eval script ==="
scp -P $port "$localRoot\628_eval_x_batch.py" "${remote}:$remoteRoot\628_eval_x_batch.py"
if ($LASTEXITCODE -ne 0) {
    Write-Host "ERROR: scp failed"
    exit 1
}
Write-Host "Synced 628_eval_x_batch.py"

# Step 2: Pause batch runner (stop schtask but don't kill training yet)
Write-Host "=== Step 2: Pausing batch runner ==="
Remote-Cmd "schtasks /End /TN sb_628_batch_runner 2>nul"
Start-Sleep -Seconds 3

# Step 3: Kill python processes (stop current training)
Write-Host "=== Step 3: Killing python processes ==="
Remote-Cmd "taskkill /F /IM python.exe"
Start-Sleep -Seconds 5

# Step 4: Run eval batch via schtasks (decoupled from SSH)
Write-Host "=== Step 4: Starting eval batch ==="
$evalBatPath = "$remoteRoot\_628_eval_runner.bat"
$evalBatContent = @"
@echo off
set PYTHON=C:\Progra~1\Python312\python.exe
set EVAL=I:\Github\Latent_Style\SchrodingerBridge\628_eval_x_batch.py
set STDOUT=I:\Github\Latent_Style\SchrodingerBridge\exp\628_ablation\destructive_logs\eval_runner_stdout.log
set STDERR=I:\Github\Latent_Style\SchrodingerBridge\exp\628_ablation\destructive_logs\eval_runner_stderr.log
cd /d I:\Github\Latent_Style\SchrodingerBridge
"%PYTHON%" "%EVAL%" > "%STDOUT%" 2> "%STDERR%"
"@
$evalBatContent | ssh -o ConnectTimeout=15 $remote -p $port "powershell -NoProfile -Command Set-Content -Path '$evalBatPath' -Value '$evalBatContent' -Encoding Default"

# Create and run eval schtask
Remote-Cmd "schtasks /Create /TN sb_628_eval_runner /TR `"$evalBatPath`" /SC ONCE /ST 23:59 /RU Administrator /IT /F 2>nul"
Remote-Cmd "schtasks /Run /TN sb_628_eval_runner"
Start-Sleep -Seconds 5

# Verify eval is running
Write-Host "=== Step 5: Verifying eval is running ==="
$tasklist = Remote-Cmd "tasklist | findstr python"
Write-Host $tasklist

Write-Host ""
Write-Host "=== DONE: Eval batch started ==="
Write-Host "Monitor with: ssh Administrator@100.115.18.62 -p 2222 'type I:\Github\Latent_Style\SchrodingerBridge\exp\628_ablation\destructive_logs\eval_batch_log.txt'"
