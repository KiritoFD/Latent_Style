param(
    [string]$Root = "I:\Github\Latent_Style\SchrodingerBridge"
)

$ErrorActionPreference = "Continue"
$logDir = Join-Path $Root "logs\remote_swd_ablation"
$expDir = Join-Path $Root "exp\remote_swd_ablation"

"== PROCESSES =="
Get-CimInstance Win32_Process |
    Where-Object {
        $cmd = [string]$_.CommandLine
        $cmd -match "remote_swd_ablation" -or
        $cmd -match "launch_remote_swd_ablation" -or
        $cmd -match "src\\run\.py"
    } |
    Select-Object ProcessId,Name,CreationDate,CommandLine |
    Format-List

"== LOG FILES =="
if (Test-Path -LiteralPath $logDir) {
    Get-ChildItem -LiteralPath $logDir |
        Sort-Object LastWriteTime -Descending |
        Select-Object Name,Length,LastWriteTime |
        Format-Table -AutoSize
} else {
    "NO_LOG_DIR $logDir"
}

"== LAUNCHER OUT =="
Get-Content -Tail 30 -LiteralPath (Join-Path $logDir "launcher.out.log") -ErrorAction SilentlyContinue

"== LAUNCHER ERR =="
Get-Content -Tail 30 -LiteralPath (Join-Path $logDir "launcher.err.log") -ErrorAction SilentlyContinue

"== LATEST TRAIN LOG =="
$latest = Get-ChildItem -LiteralPath $logDir -Filter "*.train.log" -ErrorAction SilentlyContinue |
    Sort-Object LastWriteTime -Descending |
    Select-Object -First 1
if ($latest) {
    "FILE $($latest.FullName)"
    Get-Content -Tail 50 -LiteralPath $latest.FullName
} else {
    "NO_TRAIN_LOG"
}

"== SUMMARIES =="
if (Test-Path -LiteralPath $expDir) {
    Get-ChildItem -LiteralPath $expDir -Recurse -Filter summary.json -ErrorAction SilentlyContinue |
        Select-Object FullName,Length,LastWriteTime |
        Format-Table -AutoSize
}

"== NVIDIA-SMI =="
nvidia-smi
