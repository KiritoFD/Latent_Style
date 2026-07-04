# Start the 628 destructive batch runner fully detached (survives SSH disconnect)
$root = 'I:/Github/Latent_Style/SchrodingerBridge'
$runner = "$root\628_run_destructive_batch.py"
$python = 'C:\Progra~1\Python312\python.exe'
$stdoutLog = "$root\exp\628_ablation\destructive_logs\batch_runner_stdout.log"
$stderrLog = "$root\exp\628_ablation\destructive_logs\batch_runner_stderr.log"
$pidFile = "$root\exp\628_ablation\destructive_logs\batch_runner.pid"

$logDir = Split-Path $stdoutLog -Parent
if (-not (Test-Path $logDir)) { New-Item -ItemType Directory -Path $logDir -Force | Out-Null }

# Remove old PID file if present
if (Test-Path $pidFile) { Remove-Item $pidFile -Force }

Write-Host "Starting batch runner detached..."
Write-Host "  runner : $runner"
Write-Host "  stdout : $stdoutLog"
Write-Host "  stderr : $stderrLog"

# Start-Process with -WindowStyle Hidden creates a process that is NOT a child
# of this PowerShell session (uses Windows shell create process semantics),
# so it survives SSH disconnect.
$proc = Start-Process -FilePath $python `
    -ArgumentList "`"$runner`"" `
    -WorkingDirectory $root `
    -RedirectStandardOutput $stdoutLog `
    -RedirectStandardError $stderrLog `
    -WindowStyle Hidden `
    -PassThru

Start-Sleep -Seconds 3

if ($proc -and -not $proc.HasExited) {
    $proc.Id | Out-File -FilePath $pidFile -Encoding ascii
    Write-Host "SUCCESS: Started PID=$($proc.Id)"
    Write-Host "PID file: $pidFile"
} else {
    Write-Host "FAILED to start batch runner"
    if ($proc) {
        Write-Host "ExitCode: $($proc.ExitCode)"
        if (Test-Path $stderrLog) {
            Write-Host "--- stderr ---"
            Get-Content $stderrLog -Tail 20
        }
    }
    exit 1
}

# Wait a bit more to confirm it's truly running and not crashing immediately
Start-Sleep -Seconds 5
$check = Get-Process -Id $proc.Id -ErrorAction SilentlyContinue
if ($check -and -not $check.HasExited) {
    Write-Host "Confirmed: PID=$($proc.Id) still running after 8s"
} else {
    Write-Host "WARNING: PID=$($proc.Id) exited shortly after start"
    if (Test-Path $stderrLog) {
        Write-Host "--- stderr ---"
        Get-Content $stderrLog -Tail 20
    }
}
