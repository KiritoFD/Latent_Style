param(
    [string]$Remote = "administrator@100.115.18.62",
    [int]$Port = 2222,
    [string]$RemoteRepo = "I:\Github\Latent_Style\SchrodingerBridge",
    [string]$LocalStatusRoot = "G:\GitHub\Latent_Style\SchrodingerBridge\exp\vae_backend_256_status"
)

$ErrorActionPreference = "Continue"
New-Item -ItemType Directory -Force -Path $LocalStatusRoot | Out-Null

$remoteScript = @"
Set-Location '$RemoteRepo'
Write-Output '###TIMESTAMP'
(Get-Date).ToString('o')
Write-Output '###GPU'
nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu,temperature.gpu --format=csv,noheader,nounits
Write-Output '###PROCESSES'
Get-CimInstance Win32_Process |
  Where-Object { `$_.CommandLine -match 'run_vae_backend|preprocess_latents|src\\run.py|run_evaluation.py' } |
  Select-Object ProcessId,Name,CommandLine |
  ConvertTo-Csv -NoTypeInformation
Write-Output '###TASKS'
`$taskRows = @()
foreach (`$name in @('LANCET_VAE_Backend_256_Probe','LANCET_VAE_Backend_256_SDXL')) {
  `$task = Get-ScheduledTask -TaskName `$name -ErrorAction SilentlyContinue
  if (`$null -eq `$task) {
    `$taskRows += [pscustomobject]@{task=`$name; exists='false'; state=''; last_result=''; last_run_time=''; next_run_time=''}
  } else {
    `$info = Get-ScheduledTaskInfo -TaskName `$name
    `$taskRows += [pscustomobject]@{task=`$name; exists='true'; state=[string]`$task.State; last_result=`$info.LastTaskResult; last_run_time=`$info.LastRunTime; next_run_time=`$info.NextRunTime}
  }
}
`$taskRows | ConvertTo-Csv -NoTypeInformation
Write-Output '###SDXL_CSV'
if (Test-Path 'exp\vae_backend_256_sdxl\vae_backend_256_results.csv') { Get-Content 'exp\vae_backend_256_sdxl\vae_backend_256_results.csv' -Tail 12 }
Write-Output '###PROBE_CSV'
if (Test-Path 'exp\vae_backend_256_probe\vae_backend_256_results.csv') { Get-Content 'exp\vae_backend_256_probe\vae_backend_256_results.csv' -Tail 8 }
"@

$encoded = [Convert]::ToBase64String([Text.Encoding]::Unicode.GetBytes($remoteScript))
$raw = & ssh -p $Port -o BatchMode=yes -o LogLevel=ERROR -o ConnectTimeout=8 -o ServerAliveInterval=5 -o ServerAliveCountMax=1 $Remote "powershell -NoProfile -EncodedCommand $encoded"

$txtPath = Join-Path $LocalStatusRoot "status_raw.txt"
$mdPath = Join-Path $LocalStatusRoot "status.md"
$heartbeat = Join-Path $LocalStatusRoot "heartbeat.csv"
$raw | Set-Content -Path $txtPath -Encoding UTF8

function Get-Section([string[]]$Lines, [string]$Name) {
    $start = [Array]::IndexOf($Lines, "###$Name")
    if ($start -lt 0) { return @() }
    $end = $Lines.Length
    for ($i = $start + 1; $i -lt $Lines.Length; $i++) {
        if ($Lines[$i].StartsWith("###")) { $end = $i; break }
    }
    if ($end -le ($start + 1)) { return @() }
    return @($Lines[($start + 1)..($end - 1)] | Where-Object { $_ -ne $null })
}

$lines = @($raw | Where-Object { $_ -notlike "#< CLIXML*" -and $_ -notlike "<Objs Version=*" })
$timestamp = (Get-Section $lines "TIMESTAMP" | Select-Object -First 1)
$gpu = Get-Section $lines "GPU"
$procs = Get-Section $lines "PROCESSES"
$tasks = Get-Section $lines "TASKS"
$sdxl = Get-Section $lines "SDXL_CSV"
$probe = Get-Section $lines "PROBE_CSV"

@(
    "# VAE Backend Remote Status",
    "",
    "- timestamp: $timestamp",
    "- remote: $Remote port $Port",
    "",
    "## GPU",
    ($gpu -join "`n"),
    "",
    "## Processes CSV",
    '```csv',
    ($procs -join "`n"),
    '```',
    "",
    "## Tasks CSV",
    '```csv',
    ($tasks -join "`n"),
    '```',
    "",
    "## SDXL CSV Tail",
    '```csv',
    ($sdxl -join "`n"),
    '```',
    "",
    "## Probe CSV Tail",
    '```csv',
    ($probe -join "`n"),
    '```'
) | Set-Content -Path $mdPath -Encoding UTF8

if (-not (Test-Path $heartbeat)) {
    "timestamp,gpu,sdxl_lines,probe_lines" | Set-Content -Path $heartbeat -Encoding UTF8
}
$gpuOneLine = ($gpu -join " | ").Replace(",", ";")
"$timestamp,$gpuOneLine,$(@($sdxl).Count),$(@($probe).Count)" | Add-Content -Path $heartbeat -Encoding UTF8
