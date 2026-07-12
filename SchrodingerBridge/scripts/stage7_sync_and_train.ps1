# Stage7: 远程同步代码 + 启动训练 (RTX 3060 12GB)
# 同步 src/model.py, src/config_schema.py, configs/exp_sty_stage7_delta.json 到远程
# 然后通过 SSH 启动训练 (blocking=false)
$ErrorActionPreference = "Continue"
$sshHost = "administrator@100.115.18.62"
$sshPort = "2222"
$sshOpts = @("-o", "LogLevel=ERROR", "-o", "ConnectTimeout=10")
$remoteRoot = "I:/Github/Latent_Style/SchrodingerBridge"

Write-Output "=== STAGE7 SYNC + TRAIN START $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') ==="

# ===== PHASE 1: SYNC CODE TO REMOTE =====
Write-Output "--- Syncing code to remote ---"
$filesToSync = @(
    @{local="src\model.py";         remote="$remoteRoot/src/model.py"},
    @{local="src\config_schema.py"; remote="$remoteRoot/src/config_schema.py"},
    @{local="configs\exp_sty_stage7_delta.json"; remote="$remoteRoot/configs/exp_sty_stage7_delta.json"}
)

foreach ($f in $filesToSync) {
    Write-Output "  scp $($f.local) -> $($f.remote)"
    & scp -P $sshPort @sshOpts $($f.local) "${sshHost}:$($f.remote)"
    if ($LASTEXITCODE -ne 0) {
        Write-Output "  ERROR: scp failed for $($f.local)"
    }
}

# Verify remote files
Write-Output "--- Verifying remote code ---"
& ssh -p $sshPort @sshOpts $sshHost "powershell -Command `"Get-Content $remoteRoot\src\config_schema.py | Select-String 'style_delta_head_enabled' | Select-Object -First 1`""
if ($LASTEXITCODE -ne 0) {
    Write-Output "ERROR: Remote verification failed. Code sync may have failed."
    exit 1
}

# ===== PHASE 2: START REMOTE TRAINING =====
Write-Output "--- Starting remote training ---"
$trainCmd = "cd $remoteRoot; `$env:PYTHONIOENCODING='utf-8'; python -u src\run.py --config configs\exp_sty_stage7_delta.json"
$logDir = "C:\Users\Administrator\logs\sty_inject"
$trainLog = "$logDir\stage7_delta_train.out"

# Create log dir on remote and start training via wmic (background process)
$wmicCmd = "powershell -Command `"if (-not (Test-Path '$logDir')) { New-Item -ItemType Directory -Path '$logDir' -Force | Out-Null }; `$proc = Start-Process powershell -ArgumentList '-NoProfile','-Command','cd $remoteRoot; `$env:PYTHONIOENCODING=''utf-8''; python -u src\run.py --config configs\exp_sty_stage7_delta.json 2>&1 | Tee-Object -FilePath '$trainLog'' -WindowStyle Hidden -PassThru; Write-Output `$proc.Id`""

Write-Output "  Launching training on remote..."
$remotePid = & ssh -p $sshPort @sshOpts $sshHost $wmicCmd
Write-Output "  Remote training PID: $remotePid"

if (-not $remotePid) {
    Write-Output "  WARNING: Could not get remote PID. Trying direct SSH run..."
    # Fallback: run directly (will block this terminal)
    & ssh -p $sshPort @sshOpts $sshHost "powershell -Command `"cd $remoteRoot; `$env:PYTHONIOENCODING='utf-8'; python -u src\run.py --config configs\exp_sty_stage7_delta.json 2>&1 | Tee-Object -FilePath '$trainLog'`""
} else {
    Write-Output "  Training launched in background on remote."
    Write-Output "  Log: $trainLog"
    Write-Output "  Monitor: ssh -p $sshPort $sshHost 'Get-Content $trainLog -Tail 20'"
}

Write-Output "=== STAGE7 SYNC + TRAIN DONE $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') ==="
