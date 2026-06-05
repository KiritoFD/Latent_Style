Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$Remote = 'administrator@100.115.18.62'
$Port = '2222'
$RemoteRoot = 'I:\Github\Latent_Style_TokenizerClean\SchrodingerBridge'
$LocalOut = Join-Path $PSScriptRoot 'manual_remote_tokenizerclean_trained_no_summary_third_pass_20260605.csv'

$RemoteScriptName = 'codex_tokenizerclean_trained_no_summary_third_pass.ps1'
$RemoteTempPath = "C:/Users/Administrator/AppData/Local/Temp/$RemoteScriptName"
$RemoteTempWinPath = "C:\Users\Administrator\AppData\Local\Temp\$RemoteScriptName"

$dirs = @(
    'tokenizer_t01_carrier_base_b160',
    'wikiart_distinct5_ema_lancet_spectralstat_e2_b80',
    'wikiart_distinct5_ema_lancet_spectralstat_from_e8_e16_b56',
    'wikiart512_ema_pair_budget_tokonly_e1_b80',
    'wikiart512_ema_spectral_stat_full_e2_from_tok_b48',
    'wikiart512_ema_tokenbudget_tokonly_e1_from_spectral_b48',
    'wikiart512_ema_trueint_stylepush_tsw40_kin025_e1_b48'
)

$DirLiteral = ($dirs | ConvertTo-Json -Compress)
$RemoteScript = @"
Set-StrictMode -Version Latest
`$ErrorActionPreference = 'Stop'
`$root = '$RemoteRoot'
`$dirs = '$DirLiteral' | ConvertFrom-Json

function Read-TailText {
  param([string]`$Path, [int]`$Tail = 6)
  if (-not (Test-Path -LiteralPath `$Path)) { return '' }
  return ((Get-Content -LiteralPath `$Path -Tail `$Tail -ErrorAction SilentlyContinue) -join ' || ')
}

function Short-List {
  param([object[]]`$Files, [int]`$Limit = 12)
  if (-not `$Files -or `$Files.Count -eq 0) { return '' }
  return ((`$Files | Select-Object -First `$Limit | ForEach-Object {
    `$rel = `$_.FullName.Substring(`$root.Length + 1)
    "`${rel}:`$([math]::Round([double]`$_.Length / 1MB, 6))MB"
  }) -join ' | ')
}

`$out = foreach (`$dir in `$dirs) {
  `$path = Join-Path (Join-Path `$root 'exp') `$dir
  `$exists = Test-Path -LiteralPath `$path
  if (-not `$exists) {
    [pscustomobject]@{
      remote_root = `$root
      exp_dir = `$dir
      exists = 'False'
      total_file_count = '0'
      total_mb = '0'
      weight_count = '0'
      weight_mb = '0'
      weight_files = ''
      config_count = '0'
      summary_like_count = '0'
      training_csv_count = '0'
      latest_training_csv = ''
      training_rows = ''
      last_epoch = ''
      last_loss = ''
      last_epoch_time_sec = ''
      last_samples_per_sec = ''
      log_count = '0'
      latest_log_tail = ''
      failure_marker = 'False'
      completion_marker = 'False'
      delete_whitelist = 'no'
      policy_action = 'missing_on_third_pass'
      reason = 'Directory no longer exists at checked path.'
    }
    continue
  }
  `$files = @(Get-ChildItem -LiteralPath `$path -Recurse -File -Force -ErrorAction SilentlyContinue)
  `$weights = @(`$files | Where-Object { `$_.Name -match '\.(pt|pth|ckpt|safetensors)$' })
  `$configs = @(`$files | Where-Object { `$_.Name -eq 'config.json' })
  `$summaries = @(`$files | Where-Object { `$_.Name -match 'summary.*\.(json|csv|md)$' -or `$_.FullName -match '\\full_eval\\' })
  `$training = @(`$files | Where-Object { `$_.Name -match 'training.*\.csv$' } | Sort-Object LastWriteTime -Descending)
  `$logs = @(`$files | Where-Object { `$_.Name -match '\.(log|err|out|txt)$' } | Sort-Object LastWriteTime -Descending)
  `$trainRows = @()
  `$last = `$null
  if (`$training.Count -gt 0) {
    try {
      `$trainRows = @(Import-Csv -LiteralPath `$training[0].FullName)
      `$last = `$trainRows | Select-Object -Last 1
    } catch {
      `$trainRows = @()
      `$last = `$null
    }
  }
  `$latestLogTail = if (`$logs.Count -gt 0) { Read-TailText -Path `$logs[0].FullName -Tail 8 } else { '' }
  `$allTail = ((`$logs | Select-Object -First 4 | ForEach-Object { Read-TailText -Path `$_.FullName -Tail 4 }) -join ' || ')
  `$failure = [bool](`$allTail -match '(Traceback|RuntimeError|CUDA out of memory|Exception|failed|error)')
  `$complete = [bool](`$allTail -match '(Training completed|completed|finished|epoch)' -or `$trainRows.Count -gt 0)
  [pscustomobject]@{
    remote_root = `$root
    exp_dir = `$dir
    exists = 'True'
    total_file_count = [string]`$files.Count
    total_mb = [string]([math]::Round((`$files | Measure-Object -Property Length -Sum).Sum / 1MB, 6))
    weight_count = [string]`$weights.Count
    weight_mb = [string]([math]::Round((`$weights | Measure-Object -Property Length -Sum).Sum / 1MB, 6))
    weight_files = Short-List -Files `$weights -Limit 12
    config_count = [string]`$configs.Count
    summary_like_count = [string]`$summaries.Count
    training_csv_count = [string]`$training.Count
    latest_training_csv = if (`$training.Count -gt 0) { `$training[0].FullName.Substring(`$root.Length + 1) } else { '' }
    training_rows = [string]`$trainRows.Count
    last_epoch = if (`$last) { [string]`$last.epoch } else { '' }
    last_loss = if (`$last) { [string]`$last.loss } else { '' }
    last_epoch_time_sec = if (`$last) { [string]`$last.epoch_time_sec } else { '' }
    last_samples_per_sec = if (`$last) { [string]`$last.samples_per_sec } else { '' }
    log_count = [string]`$logs.Count
    latest_log_tail = `$latestLogTail
    failure_marker = [string]`$failure
    completion_marker = [string]`$complete
    delete_whitelist = 'no'
    policy_action = 'keep_payload_pending_summary_or_owner'
    reason = 'Current third pass confirms config/training evidence and restorable weights but no summary/full_eval evidence; deleting would remove the only payload before owner decision or summary recovery.'
  }
}
`$out | ConvertTo-Csv -NoTypeInformation
"@

$LocalRemoteScript = Join-Path $env:TEMP $RemoteScriptName
$RemoteScript | Set-Content -Path $LocalRemoteScript -Encoding UTF8
& scp -P $Port -o LogLevel=ERROR $LocalRemoteScript "${Remote}:$RemoteTempPath" | Out-Null

$raw = & ssh -p $Port -o LogLevel=ERROR $Remote "powershell -NoProfile -ExecutionPolicy Bypass -File `"$RemoteTempWinPath`""
$csvLines = @($raw | Where-Object { $_ -match '^"' })
if (-not $csvLines -or $csvLines.Count -lt 2) {
    $joined = ($raw -join "`n")
    throw "Remote TokenizerClean third-pass audit did not return CSV. Output: $joined"
}
$csvLines | Set-Content -Path $LocalOut -Encoding UTF8
& ssh -p $Port -o LogLevel=ERROR $Remote "powershell -NoProfile -Command `"Remove-Item -LiteralPath '$RemoteTempWinPath' -Force -ErrorAction SilentlyContinue`"" | Out-Null
Write-Host "Wrote $LocalOut"
