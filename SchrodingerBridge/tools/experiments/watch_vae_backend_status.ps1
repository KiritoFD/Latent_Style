param(
    [string]$RepoRoot = "I:\Github\Latent_Style\SchrodingerBridge",
    [string]$StatusRoot = "I:\Github\Latent_Style\SchrodingerBridge\exp\vae_backend_256_status",
    [string]$ProbeRoot = "I:\Github\Latent_Style\SchrodingerBridge\exp\vae_backend_256_probe",
    [string]$StableRoot = "I:\Github\Latent_Style\SchrodingerBridge\exp\vae_backend_256_sdxl",
    [int]$LogTail = 80
)

$ErrorActionPreference = "Continue"
$now = Get-Date
New-Item -ItemType Directory -Force -Path $StatusRoot | Out-Null

function Get-GpuStatus {
    try {
        $raw = & nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu,temperature.gpu,power.draw --format=csv,noheader,nounits 2>$null
        return @($raw | ForEach-Object {
            $parts = $_.Split(",") | ForEach-Object { $_.Trim() }
            [pscustomobject]@{
                name = $parts[0]
                memory_used_mb = [int]$parts[1]
                memory_total_mb = [int]$parts[2]
                utilization_gpu_pct = [int]$parts[3]
                temperature_c = [int]$parts[4]
                power_w = $parts[5]
            }
        })
    } catch {
        return @([pscustomobject]@{ error = $_.Exception.Message })
    }
}

function Get-MatchingProcesses {
    try {
        return @(Get-CimInstance Win32_Process |
            Where-Object { $_.CommandLine -match "run_vae_backend|preprocess_latents|src\\run.py|run_evaluation.py" } |
            Select-Object ProcessId, Name, CommandLine)
    } catch {
        return @([pscustomobject]@{ error = $_.Exception.Message })
    }
}

function Get-TaskRows {
    $names = @(
        "LANCET_VAE_Backend_256_Probe",
        "LANCET_VAE_Backend_256_SDXL",
        "LANCET_VAE_Backend_StatusWatch"
    )
    $rows = @()
    foreach ($name in $names) {
        try {
            $task = Get-ScheduledTask -TaskName $name -ErrorAction SilentlyContinue
            if ($null -eq $task) {
                $rows += [pscustomobject]@{ task = $name; exists = $false; state = ""; last_result = ""; last_run_time = ""; next_run_time = "" }
                continue
            }
            $info = Get-ScheduledTaskInfo -TaskName $name
            $rows += [pscustomobject]@{
                task = $name
                exists = $true
                state = [string]$task.State
                last_result = $info.LastTaskResult
                last_run_time = $info.LastRunTime
                next_run_time = $info.NextRunTime
            }
        } catch {
            $rows += [pscustomobject]@{ task = $name; exists = $false; error = $_.Exception.Message }
        }
    }
    return $rows
}

function Read-CsvTail([string]$Path) {
    if (-not (Test-Path $Path)) {
        return @()
    }
    try {
        $rows = Import-Csv $Path
        return @($rows | Select-Object -Last 12)
    } catch {
        return @([pscustomobject]@{ error = $_.Exception.Message; path = $Path })
    }
}

function Read-LogTail([string]$Root) {
    $items = @()
    foreach ($log in Get-ChildItem -Path $Root -Filter "run.log" -Recurse -ErrorAction SilentlyContinue) {
        try {
            $items += [pscustomobject]@{
                path = $log.FullName
                last_write_time = $log.LastWriteTime
                tail = @(Get-Content $log.FullName -Tail $LogTail -ErrorAction SilentlyContinue)
            }
        } catch {
            $items += [pscustomobject]@{ path = $log.FullName; error = $_.Exception.Message }
        }
    }
    return @($items | Sort-Object last_write_time -Descending | Select-Object -First 6)
}

$probeCsv = Join-Path $ProbeRoot "vae_backend_256_results.csv"
$stableCsv = Join-Path $StableRoot "vae_backend_256_results.csv"
$payload = [ordered]@{
    timestamp = $now.ToString("o")
    repo_root = $RepoRoot
    gpu = Get-GpuStatus
    tasks = Get-TaskRows
    processes = Get-MatchingProcesses
    results = [ordered]@{
        probe_csv = $probeCsv
        probe_tail = Read-CsvTail $probeCsv
        sdxl_csv = $stableCsv
        sdxl_tail = Read-CsvTail $stableCsv
    }
    logs = [ordered]@{
        probe = Read-LogTail $ProbeRoot
        sdxl = Read-LogTail $StableRoot
    }
}

$jsonPath = Join-Path $StatusRoot "status.json"
$mdPath = Join-Path $StatusRoot "status.md"
$payload | ConvertTo-Json -Depth 8 | Set-Content -Path $jsonPath -Encoding UTF8

$gpuLine = ($payload.gpu | ForEach-Object {
    if ($_.error) { "GPU: $($_.error)" } else { "GPU: $($_.name), $($_.memory_used_mb)/$($_.memory_total_mb) MB, util=$($_.utilization_gpu_pct)%, temp=$($_.temperature_c)C" }
}) -join "`n"
$procCount = @($payload.processes).Count
$probeCount = @($payload.results.probe_tail).Count
$sdxlCount = @($payload.results.sdxl_tail).Count
@(
    "# VAE Backend Status",
    "",
    "- timestamp: $($payload.timestamp)",
    "- process_count: $procCount",
    "- probe_rows_tail: $probeCount",
    "- sdxl_rows_tail: $sdxlCount",
    "",
    "## GPU",
    $gpuLine,
    "",
    "## Tasks",
    ($payload.tasks | Format-Table -AutoSize | Out-String),
    "",
    "## Processes",
    ($payload.processes | Select-Object ProcessId,Name,CommandLine | Format-Table -Wrap | Out-String)
) | Set-Content -Path $mdPath -Encoding UTF8

$heartbeat = Join-Path $StatusRoot "heartbeat.csv"
if (-not (Test-Path $heartbeat)) {
    "timestamp,gpu_memory_used_mb,gpu_memory_total_mb,gpu_utilization_pct,process_count,probe_tail_rows,sdxl_tail_rows" | Set-Content -Path $heartbeat -Encoding UTF8
}
$firstGpu = @($payload.gpu | Select-Object -First 1)[0]
$memUsed = if ($firstGpu.memory_used_mb -ne $null) { $firstGpu.memory_used_mb } else { "" }
$memTotal = if ($firstGpu.memory_total_mb -ne $null) { $firstGpu.memory_total_mb } else { "" }
$util = if ($firstGpu.utilization_gpu_pct -ne $null) { $firstGpu.utilization_gpu_pct } else { "" }
"$($payload.timestamp),$memUsed,$memTotal,$util,$procCount,$probeCount,$sdxlCount" | Add-Content -Path $heartbeat -Encoding UTF8
