Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$Remote = 'administrator@100.115.18.62'
$Port = '2222'
$CleanupDir = Join-Path $PSScriptRoot 'cleanup'
if (-not (Test-Path -LiteralPath $CleanupDir)) {
    New-Item -ItemType Directory -Path $CleanupDir | Out-Null
}

$LocalLedger = Join-Path $CleanupDir 'manual_remote_tokenizerclean_orphan_probe_weight_cleanup_20260605.csv'
$LocalVerify = Join-Path $PSScriptRoot 'manual_remote_tokenizerclean_orphan_probe_post_delete_verify_20260605.csv'

$RemoteScriptName = 'codex_delete_tokenizerclean_orphan_probe_weights.ps1'
$RemoteLedgerName = 'codex_tokenizerclean_orphan_probe_weight_cleanup.csv'
$RemoteVerifyName = 'codex_tokenizerclean_orphan_probe_verify.csv'
$RemoteTempPath = "C:/Users/Administrator/AppData/Local/Temp/$RemoteScriptName"
$RemoteTempWinPath = "C:\Users\Administrator\AppData\Local\Temp\$RemoteScriptName"
$RemoteLedgerPath = "C:/Users/Administrator/AppData/Local/Temp/$RemoteLedgerName"
$RemoteVerifyPath = "C:/Users/Administrator/AppData/Local/Temp/$RemoteVerifyName"
$RemoteLedgerWinPath = "C:\Users\Administrator\AppData\Local\Temp\$RemoteLedgerName"
$RemoteVerifyWinPath = "C:\Users\Administrator\AppData\Local\Temp\$RemoteVerifyName"

$RemoteScript = @'
Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$root = 'I:\Github\Latent_Style_TokenizerClean\SchrodingerBridge'
$ledgerPath = 'C:\Users\Administrator\AppData\Local\Temp\codex_tokenizerclean_orphan_probe_weight_cleanup.csv'
$verifyPath = 'C:\Users\Administrator\AppData\Local\Temp\codex_tokenizerclean_orphan_probe_verify.csv'

$fileTargets = @(
  'exp\axis_scale_probe\color_x150.pt',
  'exp\axis_scale_probe\color_x250.pt',
  'exp\axis_scale_probe\geometry_x125.pt',
  'exp\axis_scale_probe\specgeo_x120.pt',
  'exp\axis_scale_probe\spectrum_x125.pt',
  'exp\axis_scale_probe\spectrum_x175.pt',
  'exp\field_budget_release_probe\release_s100.pt',
  'exp\field_budget_release_probe\release_s150.pt',
  'exp\pair_relative_geometry_release_probe\pairrel_c090_g100_b040_from_spatial_gate.pt',
  'exp\pair_relative_geometry_release_probe\pairrel_c120_g050_b020_from_spatial_gate.pt',
  'exp\pair_relative_geometry_release_probe\pairrel_c120_g050_b020_spgain013_from_spatial_gate.pt'
)

$dirTargets = @(
  'exp\axis_scale_probe',
  'exp\field_budget_release_probe',
  'exp\pair_relative_geometry_release_probe'
)

function Resolve-RemoteTarget {
  param([string]$RelativePath)
  $candidate = Join-Path $root $RelativePath
  if (-not (Test-Path -LiteralPath $candidate)) { return $null }
  $resolved = (Resolve-Path -LiteralPath $candidate).Path
  $prefix = $root.TrimEnd('\') + '\'
  if (-not $resolved.StartsWith($prefix, [StringComparison]::OrdinalIgnoreCase)) {
    throw "Refusing path outside remote root: $resolved"
  }
  return $resolved
}

$ledger = New-Object System.Collections.Generic.List[object]

foreach ($rel in $fileTargets) {
  $full = Resolve-RemoteTarget $rel
  $existsBefore = $null -ne $full
  $beforeBytes = 0
  $beforeMb = 0
  $lastWrite = ''
  $status = 'already_absent'
  $postExists = $false
  if ($existsBefore) {
    $item = Get-Item -LiteralPath $full -Force
    if ($item.PSIsContainer) { throw "Expected file but found directory: $rel" }
    $beforeBytes = $item.Length
    $beforeMb = [math]::Round([double]$beforeBytes / 1MB, 6)
    $lastWrite = $item.LastWriteTime
    Remove-Item -LiteralPath $full -Force
    $postExists = Test-Path -LiteralPath $full
    if ($postExists) { throw "Deletion failed for $rel" }
    $status = 'deleted'
  }
  $ledger.Add([pscustomobject]@{
    cleanup_run = (Get-Date).ToString('s')
    remote_root = $root
    relative_path = $rel
    target_type = 'file'
    cleanup_class = 'tokenizerclean_orphan_probe_weight'
    exists_before = $existsBefore
    before_bytes = $beforeBytes
    before_mb = $beforeMb
    last_write_time = $lastWrite
    status = $status
    post_exists = $postExists
    reason = '0 citation, no config/log/training csv/summary, diagnostics output retained'
  })
}

foreach ($rel in $dirTargets) {
  $full = Resolve-RemoteTarget $rel
  $existsBefore = $null -ne $full
  $childCount = ''
  $status = 'already_absent'
  $postExists = $false
  if ($existsBefore) {
    $children = @(Get-ChildItem -LiteralPath $full -Force -ErrorAction SilentlyContinue)
    $childCount = $children.Count
    if ($childCount -eq 0) {
      Remove-Item -LiteralPath $full -Force
      $postExists = Test-Path -LiteralPath $full
      if ($postExists) { throw "Directory deletion failed for $rel" }
      $status = 'deleted_empty_dir'
    } else {
      $postExists = $true
      $status = "kept_nonempty_dir_child_count_$childCount"
    }
  }
  $ledger.Add([pscustomobject]@{
    cleanup_run = (Get-Date).ToString('s')
    remote_root = $root
    relative_path = $rel
    target_type = 'directory'
    cleanup_class = 'tokenizerclean_empty_orphan_probe_dir'
    exists_before = $existsBefore
    before_bytes = 0
    before_mb = 0
    last_write_time = ''
    status = $status
    post_exists = $postExists
    reason = 'remove only after exact orphan weights were deleted and directory is empty'
  })
}

$verifyTargets = @(
  [pscustomobject]@{ relative_path='exp\axis_scale_probe'; expected='absent'; reason='orphan probe dir deleted' },
  [pscustomobject]@{ relative_path='exp\field_budget_release_probe'; expected='absent'; reason='orphan probe dir deleted' },
  [pscustomobject]@{ relative_path='exp\pair_relative_geometry_release_probe'; expected='absent'; reason='orphan probe dir deleted' },
  [pscustomobject]@{ relative_path='exp\diagnostics\axis_scale_probe_n6_summary.json'; expected='present'; reason='diagnostics summary retained' },
  [pscustomobject]@{ relative_path='exp\diagnostics\axis_scale_color_x150_n6'; expected='present'; reason='diagnostics output retained' },
  [pscustomobject]@{ relative_path='exp\diagnostics\field_budget_release_release_s100_n6'; expected='present'; reason='diagnostics output retained' },
  [pscustomobject]@{ relative_path='exp\diagnostics\pairrel_c090_g100_b040_from_spatial_gate_n6'; expected='present'; reason='diagnostics output retained' },
  [pscustomobject]@{ relative_path='exp\diagnostics\pairrel_c120_g050_b020_from_spatial_gate_n6'; expected='present'; reason='diagnostics output retained' },
  [pscustomobject]@{ relative_path='exp\tokenizer_t01_carrier_base_b160'; expected='present'; reason='trained no-summary payload retained' },
  [pscustomobject]@{ relative_path='exp\wikiart_distinct5_ema_lancet_spectralstat_e2_b80'; expected='present'; reason='trained no-summary payload retained' },
  [pscustomobject]@{ relative_path='exp\wikiart512_ema_spectral_stat_full_e2_from_tok_b48'; expected='present'; reason='trained no-summary payload retained' }
)

$verify = foreach ($target in $verifyTargets) {
  $path = Join-Path $root $target.relative_path
  $exists = Test-Path -LiteralPath $path
  [pscustomobject]@{
    verify_run = (Get-Date).ToString('s')
    remote_root = $root
    relative_path = $target.relative_path
    expected = $target.expected
    exists = $exists
    pass = (($target.expected -eq 'present' -and $exists) -or ($target.expected -eq 'absent' -and -not $exists))
    reason = $target.reason
  }
}

$ledger | Export-Csv -NoTypeInformation -Encoding UTF8 -Path $ledgerPath
$verify | Export-Csv -NoTypeInformation -Encoding UTF8 -Path $verifyPath
'@

$LocalRemoteScript = Join-Path $env:TEMP $RemoteScriptName
$RemoteScript | Set-Content -Path $LocalRemoteScript -Encoding UTF8
& scp -P $Port -o LogLevel=ERROR $LocalRemoteScript "${Remote}:$RemoteTempPath" | Out-Null
& ssh -p $Port -o LogLevel=ERROR $Remote "powershell -NoProfile -ExecutionPolicy Bypass -File `"$RemoteTempWinPath`"" | Out-Null
& scp -P $Port -o LogLevel=ERROR "${Remote}:$RemoteLedgerPath" $LocalLedger | Out-Null
& scp -P $Port -o LogLevel=ERROR "${Remote}:$RemoteVerifyPath" $LocalVerify | Out-Null
& ssh -p $Port -o LogLevel=ERROR $Remote "powershell -NoProfile -Command `"Remove-Item -LiteralPath '$RemoteTempWinPath','$RemoteLedgerWinPath','$RemoteVerifyWinPath' -Force -ErrorAction SilentlyContinue`"" | Out-Null

Write-Host "Wrote $LocalLedger"
Write-Host "Wrote $LocalVerify"
