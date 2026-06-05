Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$Remote = 'administrator@100.115.18.62'
$Port = '2222'
$CleanupDir = Join-Path $PSScriptRoot 'cleanup'
if (-not (Test-Path -LiteralPath $CleanupDir)) {
    New-Item -ItemType Directory -Path $CleanupDir | Out-Null
}

$LocalLedger = Join-Path $CleanupDir 'manual_remote_rar_weight_only_archive_cleanup_20260605.csv'
$LocalVerify = Join-Path $PSScriptRoot 'manual_remote_rar_weight_only_archive_post_delete_verify_20260605.csv'

$RemoteScriptName = 'codex_delete_remote_rar_weight_only_archives.ps1'
$RemoteTempPath = "C:/Users/Administrator/AppData/Local/Temp/$RemoteScriptName"
$RemoteTempWinPath = "C:\Users\Administrator\AppData\Local\Temp\$RemoteScriptName"
$RemoteLedgerPath = 'C:/Users/Administrator/AppData/Local/Temp/codex_rar_weight_only_cleanup.csv'
$RemoteLedgerWinPath = 'C:\Users\Administrator\AppData\Local\Temp\codex_rar_weight_only_cleanup.csv'
$RemoteVerifyPath = 'C:/Users/Administrator/AppData/Local/Temp/codex_rar_weight_only_verify.csv'
$RemoteVerifyWinPath = 'C:\Users\Administrator\AppData\Local\Temp\codex_rar_weight_only_verify.csv'

$RemoteScript = @'
Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$root = 'I:\Github\Latent_Style'
$ledgerPath = 'C:\Users\Administrator\AppData\Local\Temp\codex_rar_weight_only_cleanup.csv'
$verifyPath = 'C:\Users\Administrator\AppData\Local\Temp\codex_rar_weight_only_verify.csv'

$targets = @(
  [pscustomobject]@{ rel='Cycle-NCE\Gate.rar'; group='Cycle-NCE/Gate.rar'; reason='nonweight entries all same-size existing; only old checkpoint/tokenizer weights unique' },
  [pscustomobject]@{ rel='Cycle-NCE\Attn_48.part1.rar'; group='Cycle-NCE/Attn_48.part*.rar'; reason='nonweight entries all same-size existing; only old checkpoint/tokenizer weights unique' },
  [pscustomobject]@{ rel='Cycle-NCE\Attn_48.part2.rar'; group='Cycle-NCE/Attn_48.part*.rar'; reason='nonweight entries all same-size existing; only old checkpoint/tokenizer weights unique' },
  [pscustomobject]@{ rel='Cycle-NCE\Attn_48.part3.rar'; group='Cycle-NCE/Attn_48.part*.rar'; reason='nonweight entries all same-size existing; only old checkpoint/tokenizer weights unique' },
  [pscustomobject]@{ rel='Cycle-NCE\chess.part1.rar'; group='Cycle-NCE/chess.part*.rar'; reason='nonweight entries all same-size existing; only old checkpoint/tokenizer weights unique' },
  [pscustomobject]@{ rel='Cycle-NCE\chess.part2.rar'; group='Cycle-NCE/chess.part*.rar'; reason='nonweight entries all same-size existing; only old checkpoint/tokenizer weights unique' }
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

$ledger = foreach ($target in $targets) {
  $resolved = Resolve-RemoteTarget -RelativePath $target.rel
  $existsBefore = $null -ne $resolved
  $beforeBytes = 0
  $beforeMb = 0
  $lastWrite = ''
  $status = 'already_absent'
  $postExists = $false
  if ($existsBefore) {
    $item = Get-Item -LiteralPath $resolved -Force
    if ($item.PSIsContainer) { throw "Expected archive file but found directory: $($target.rel)" }
    $beforeBytes = $item.Length
    $beforeMb = [math]::Round([double]$beforeBytes / 1MB, 6)
    $lastWrite = $item.LastWriteTime
    Remove-Item -LiteralPath $resolved -Force
    $postExists = Test-Path -LiteralPath $resolved
    if ($postExists) { throw "Deletion failed: $($target.rel)" }
    $status = 'deleted'
  }
  [pscustomobject]@{
    cleanup_run = (Get-Date).ToString('s')
    remote_root = $root
    archive_group = $target.group
    relative_path = $target.rel
    exists_before = $existsBefore
    before_bytes = $beforeBytes
    before_mb = $beforeMb
    last_write_time = $lastWrite
    status = $status
    post_exists = $postExists
    reason = $target.reason
  }
}

$verifyTargets = @(
  [pscustomobject]@{ rel='Cycle-NCE\Gate.rar'; expected='absent'; reason='deleted weight-only archive' },
  [pscustomobject]@{ rel='Cycle-NCE\Attn_48.part1.rar'; expected='absent'; reason='deleted weight-only multipart archive' },
  [pscustomobject]@{ rel='Cycle-NCE\Attn_48.part2.rar'; expected='absent'; reason='deleted weight-only multipart archive' },
  [pscustomobject]@{ rel='Cycle-NCE\Attn_48.part3.rar'; expected='absent'; reason='deleted weight-only multipart archive' },
  [pscustomobject]@{ rel='Cycle-NCE\chess.part1.rar'; expected='absent'; reason='deleted weight-only multipart archive' },
  [pscustomobject]@{ rel='Cycle-NCE\chess.part2.rar'; expected='absent'; reason='deleted weight-only multipart archive' },
  [pscustomobject]@{ rel='Cycle-NCE\Gate'; expected='present'; reason='expanded Gate evidence retained' },
  [pscustomobject]@{ rel='Cycle-NCE\Attn_48'; expected='present'; reason='expanded Attn_48 evidence retained' },
  [pscustomobject]@{ rel='Cycle-NCE\chess'; expected='present'; reason='expanded chess evidence retained' },
  [pscustomobject]@{ rel='experiments.rar'; expected='present'; reason='retained due cache mismatch' },
  [pscustomobject]@{ rel='Cycle-NCE\45.rar'; expected='present'; reason='retained unique historical archive' }
)

$verify = foreach ($target in $verifyTargets) {
  $path = Join-Path $root $target.rel
  $exists = Test-Path -LiteralPath $path
  [pscustomobject]@{
    verify_run = (Get-Date).ToString('s')
    remote_root = $root
    relative_path = $target.rel
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
