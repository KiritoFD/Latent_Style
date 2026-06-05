Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$OutPath = Join-Path $PSScriptRoot 'manual_remote_tokenizerclean_retained_no_summary_owner_review_20260605.csv'
$Remote = 'administrator@100.115.18.62'
$Port = '2222'

$RemoteScript = @'
$ErrorActionPreference = 'Stop'
$root = 'I:\Github\Latent_Style_TokenizerClean\SchrodingerBridge'
$expRoot = Join-Path $root 'exp'
$dirs = @(
  'axis_scale_probe',
  'field_budget_release_probe',
  'pair_relative_geometry_release_probe',
  'tokenizer_t01_carrier_base_b160',
  'wikiart_distinct5_ema_lancet_spectralstat_e2_b80',
  'wikiart_distinct5_ema_lancet_spectralstat_from_e8_e16_b56',
  'wikiart512_ema_pair_budget_tokonly_e1_b80',
  'wikiart512_ema_spectral_stat_full_e2_from_tok_b48',
  'wikiart512_ema_tokenbudget_tokonly_e1_from_spectral_b48',
  'wikiart512_ema_trueint_stylepush_tsw40_kin025_e1_b48'
)

function Join-Sample {
  param([object[]]$Items, [int]$Limit = 12)
  if (-not $Items -or $Items.Count -eq 0) { return '' }
  return (($Items | Select-Object -First $Limit) -join ' | ')
}

function Compact-Text {
  param([string[]]$Lines, [int]$Limit = 12)
  if (-not $Lines) { return '' }
  return (($Lines | Select-Object -First $Limit) -replace '"','''') -join ' || '
}

$rows = foreach ($dir in $dirs) {
  $path = Join-Path $expRoot $dir
  $exists = Test-Path -LiteralPath $path
  $files = @()
  $weights = @()
  $configs = @()
  $summaries = @()
  $logs = @()
  $trainingCsvs = @()
  $immediate = @()
  $latestLogTail = ''
  $latestCsvTail = ''
  $configHead = ''
  $hasFailure = $false
  $hasCompletion = $false
  $hasTraceback = $false
  $totalMb = 0
  $weightMb = 0

  if ($exists) {
    $children = @(Get-ChildItem -LiteralPath $path -Force -Recurse -File -ErrorAction SilentlyContinue)
    $files = $children
    $totalBytes = ($files | Measure-Object -Property Length -Sum).Sum
    if ($null -eq $totalBytes) { $totalBytes = 0 }
    $totalMb = [math]::Round([double]$totalBytes / 1MB, 6)

    $weights = @($files | Where-Object { $_.Extension -in @('.pt','.pth','.ckpt','.safetensors') })
    $weightBytes = ($weights | Measure-Object -Property Length -Sum).Sum
    if ($null -eq $weightBytes) { $weightBytes = 0 }
    $weightMb = [math]::Round([double]$weightBytes / 1MB, 6)

    $configs = @($files | Where-Object { $_.Name -match 'config|args|hparams|params' -and $_.Extension -in @('.json','.yaml','.yml','.txt') })
    $summaries = @($files | Where-Object { $_.Name -match 'summary|metrics|eval|result|curve|report' -and $_.Extension -in @('.json','.csv','.md','.txt') })
    $logs = @($files | Where-Object { $_.Extension -in @('.log','.out','.txt') -or $_.Name -match 'log|train' })
    $trainingCsvs = @($files | Where-Object { $_.Extension -eq '.csv' -and $_.Name -match 'train|loss|history|metric' })
    $immediate = @(Get-ChildItem -LiteralPath $path -Force -ErrorAction SilentlyContinue | Sort-Object Name | ForEach-Object {
      if ($_.PSIsContainer) { "dir:$($_.Name)" } else { "file:$($_.Name):$([math]::Round($_.Length/1MB,3))MB" }
    })

    $latestLog = $logs | Sort-Object LastWriteTime -Descending | Select-Object -First 1
    if ($latestLog) {
      $tail = @(Get-Content -LiteralPath $latestLog.FullName -Tail 20 -ErrorAction SilentlyContinue)
      $latestLogTail = Compact-Text $tail 20
      $joined = ($tail -join "`n")
      $hasFailure = [bool]($joined -match 'error|failed|traceback|exception|nan|oom|out of memory')
      $hasTraceback = [bool]($joined -match 'Traceback|Exception')
      $hasCompletion = [bool]($joined -match 'finished|completed|done|training complete|saving|epoch')
    }

    $latestCsv = $trainingCsvs | Sort-Object LastWriteTime -Descending | Select-Object -First 1
    if ($latestCsv) {
      $latestCsvTail = Compact-Text (@(Get-Content -LiteralPath $latestCsv.FullName -Tail 8 -ErrorAction SilentlyContinue)) 8
    }

    $firstConfig = $configs | Sort-Object LastWriteTime -Descending | Select-Object -First 1
    if ($firstConfig) {
      $configHead = Compact-Text (@(Get-Content -LiteralPath $firstConfig.FullName -TotalCount 16 -ErrorAction SilentlyContinue)) 16
    }
  }

  [pscustomobject]@{
    remote_root = $root
    exp_dir = $dir
    exists = $exists
    total_file_count = $files.Count
    total_mb = $totalMb
    weight_count = $weights.Count
    weight_mb = $weightMb
    weight_files = Join-Sample (@($weights | Sort-Object Name | ForEach-Object { "$($_.Name):$([math]::Round($_.Length/1MB,3))MB" })) 20
    immediate_entries = Join-Sample $immediate 30
    config_count = $configs.Count
    config_files = Join-Sample (@($configs | Sort-Object Name | ForEach-Object { $_.FullName.Substring($root.Length + 1) })) 10
    config_head = $configHead
    summary_like_count = $summaries.Count
    summary_like_files = Join-Sample (@($summaries | Sort-Object Name | ForEach-Object { $_.FullName.Substring($root.Length + 1) })) 20
    log_count = $logs.Count
    latest_log = if ($logs.Count) { ($logs | Sort-Object LastWriteTime -Descending | Select-Object -First 1).FullName.Substring($root.Length + 1) } else { '' }
    latest_log_tail = $latestLogTail
    log_has_failure_marker = $hasFailure
    log_has_traceback = $hasTraceback
    log_has_completion_marker = $hasCompletion
    training_csv_count = $trainingCsvs.Count
    latest_training_csv = if ($trainingCsvs.Count) { ($trainingCsvs | Sort-Object LastWriteTime -Descending | Select-Object -First 1).FullName.Substring($root.Length + 1) } else { '' }
    latest_training_csv_tail = $latestCsvTail
  }
}

$rows | ConvertTo-Csv -NoTypeInformation
'@

$RemoteScriptName = 'codex_inspect_tokenizerclean_retained_no_summary.ps1'
$RemoteTempPath = "C:/Users/Administrator/AppData/Local/Temp/$RemoteScriptName"
$RemoteTempWinPath = "C:\Users\Administrator\AppData\Local\Temp\$RemoteScriptName"
$LocalRemoteScript = Join-Path $env:TEMP $RemoteScriptName

$RemoteScript | Set-Content -Path $LocalRemoteScript -Encoding UTF8
& scp -P $Port -o LogLevel=ERROR $LocalRemoteScript "${Remote}:$RemoteTempPath" | Out-Null

$output = & ssh -p $Port -o LogLevel=ERROR $Remote "powershell -NoProfile -ExecutionPolicy Bypass -File `"$RemoteTempWinPath`""
if (-not $output -or $output[0] -notmatch '^"remote_root"') {
    $joined = ($output -join "`n")
    throw "Remote inspection did not return CSV. Output: $joined"
}

$output | Set-Content -Path $OutPath -Encoding UTF8
& ssh -p $Port -o LogLevel=ERROR $Remote "powershell -NoProfile -Command `"Remove-Item -LiteralPath '$RemoteTempWinPath' -Force -ErrorAction SilentlyContinue`"" | Out-Null
Write-Host "Wrote $OutPath"
