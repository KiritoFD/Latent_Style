Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$Remote = 'administrator@100.115.18.62'
$Port = '2222'
$RemoteRoot = 'I:\Github\Latent_Style'
$LocalRoot = 'G:\GitHub\Latent_Style'
$LocalOut = Join-Path $PSScriptRoot 'manual_remote_experiments_rar_symlink_targets_20260605.csv'

$RemoteScriptName = 'codex_inspect_experiments_rar_symlink_targets.ps1'
$RemoteTempPath = "C:/Users/Administrator/AppData/Local/Temp/$RemoteScriptName"
$RemoteTempWinPath = "C:\Users\Administrator\AppData\Local\Temp\$RemoteScriptName"

$entries = @(
    @{ entry = 'experiments\eval_cache\hf\models--openai--clip-vit-base-patch32\snapshots\3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268\config.json'; archive_bytes = 4186 },
    @{ entry = 'experiments\eval_cache\hf\models--openai--clip-vit-base-patch32\snapshots\3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268\merges.txt'; archive_bytes = 524657 },
    @{ entry = 'experiments\eval_cache\hf\models--openai--clip-vit-base-patch32\snapshots\3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268\preprocessor_config.json'; archive_bytes = 316 },
    @{ entry = 'experiments\eval_cache\hf\models--openai--clip-vit-base-patch32\snapshots\3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268\pytorch_model.bin'; archive_bytes = 605247071 },
    @{ entry = 'experiments\eval_cache\hf\models--openai--clip-vit-base-patch32\snapshots\3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268\special_tokens_map.json'; archive_bytes = 389 },
    @{ entry = 'experiments\eval_cache\hf\models--openai--clip-vit-base-patch32\snapshots\3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268\tokenizer.json'; archive_bytes = 2224041 },
    @{ entry = 'experiments\eval_cache\hf\models--openai--clip-vit-base-patch32\snapshots\3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268\tokenizer_config.json'; archive_bytes = 592 },
    @{ entry = 'experiments\eval_cache\hf\models--openai--clip-vit-base-patch32\snapshots\3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268\vocab.json'; archive_bytes = 862328 },
    @{ entry = 'experiments\eval_cache\hf\models--openai--clip-vit-base-patch32\snapshots\c237dc49a33fc61debc9276459120b7eac67e7ef\model.safetensors'; archive_bytes = 605157884 }
)

function Get-LocalTargetFact {
    param(
        [string]$Path,
        [Int64]$ArchiveBytes
    )
    if (-not (Test-Path -LiteralPath $Path)) {
        return [pscustomobject]@{ exists='False'; bytes=''; attributes=''; link_type=''; target=''; target_path=''; target_exists='False'; target_bytes=''; target_same_size_as_archive='False' }
    }
    $item = Get-Item -LiteralPath $Path -Force
    $target = if ($null -ne $item.Target) { [string]($item.Target -join ';') } else { '' }
    $targetPath = ''
    $targetExists = 'False'
    $targetBytes = ''
    $targetSame = 'False'
    if (-not [string]::IsNullOrWhiteSpace($target)) {
        $firstTarget = ($target -split ';')[0]
        if ([System.IO.Path]::IsPathRooted($firstTarget)) {
            $targetPath = $firstTarget
        } else {
            $targetPath = [System.IO.Path]::GetFullPath((Join-Path (Split-Path -Path $Path -Parent) $firstTarget))
        }
        if (Test-Path -LiteralPath $targetPath) {
            $targetItem = Get-Item -LiteralPath $targetPath -Force
            $targetExists = 'True'
            $targetBytes = [string][Int64]$targetItem.Length
            $targetSame = [string]([Int64]$targetItem.Length -eq $ArchiveBytes)
        }
    }
    return [pscustomobject]@{
        exists = 'True'
        bytes = [string][Int64]$item.Length
        attributes = [string]$item.Attributes
        link_type = [string]$item.LinkType
        target = $target
        target_path = $targetPath
        target_exists = $targetExists
        target_bytes = $targetBytes
        target_same_size_as_archive = $targetSame
    }
}

$RemoteEntriesLiteral = ($entries | ConvertTo-Json -Depth 3 -Compress)
$RemoteScript = @"
Set-StrictMode -Version Latest
`$ErrorActionPreference = 'Stop'
`$root = '$RemoteRoot'
`$entries = '$RemoteEntriesLiteral' | ConvertFrom-Json

function Get-TargetFact {
  param([string]`$Path, [Int64]`$ArchiveBytes)
  if (-not (Test-Path -LiteralPath `$Path)) {
    return [pscustomobject]@{ exists='False'; bytes=''; attributes=''; link_type=''; target=''; target_path=''; target_exists='False'; target_bytes=''; target_same_size_as_archive='False' }
  }
  `$item = Get-Item -LiteralPath `$Path -Force
  `$target = if (`$null -ne `$item.Target) { [string](`$item.Target -join ';') } else { '' }
  `$targetPath = ''
  `$targetExists = 'False'
  `$targetBytes = ''
  `$targetSame = 'False'
  if (-not [string]::IsNullOrWhiteSpace(`$target)) {
    `$firstTarget = (`$target -split ';')[0]
    if ([System.IO.Path]::IsPathRooted(`$firstTarget)) {
      `$targetPath = `$firstTarget
    } else {
      `$targetPath = [System.IO.Path]::GetFullPath((Join-Path (Split-Path -Path `$Path -Parent) `$firstTarget))
    }
    if (Test-Path -LiteralPath `$targetPath) {
      `$targetItem = Get-Item -LiteralPath `$targetPath -Force
      `$targetExists = 'True'
      `$targetBytes = [string][Int64]`$targetItem.Length
      `$targetSame = [string]([Int64]`$targetItem.Length -eq `$ArchiveBytes)
    }
  }
  return [pscustomobject]@{
    exists = 'True'
    bytes = [string][Int64]`$item.Length
    attributes = [string]`$item.Attributes
    link_type = [string]`$item.LinkType
    target = `$target
    target_path = `$targetPath
    target_exists = `$targetExists
    target_bytes = `$targetBytes
    target_same_size_as_archive = `$targetSame
  }
}

`$out = foreach (`$e in `$entries) {
  `$entry = [string]`$e.entry
  `$archiveBytes = [Int64]`$e.archive_bytes
  `$path = Join-Path `$root `$entry
  `$fact = Get-TargetFact -Path `$path -ArchiveBytes `$archiveBytes
  [pscustomobject]@{
    archive = 'experiments.rar'
    archive_entry = `$entry
    archive_entry_bytes = [string]`$archiveBytes
    archive_entry_mb = [string]([math]::Round([double]`$archiveBytes / 1MB, 6))
    remote_expanded_path = `$path
    remote_expanded_exists = `$fact.exists
    remote_expanded_bytes = `$fact.bytes
    remote_expanded_attributes = `$fact.attributes
    remote_expanded_link_type = `$fact.link_type
    remote_expanded_target = `$fact.target
    remote_expanded_target_path = `$fact.target_path
    remote_expanded_target_exists = `$fact.target_exists
    remote_expanded_target_bytes = `$fact.target_bytes
    remote_expanded_target_same_size = `$fact.target_same_size_as_archive
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
    throw "Remote experiments.rar symlink target audit did not return CSV. Output: $joined"
}

$remoteRows = $csvLines | ConvertFrom-Csv
$rows = foreach ($row in $remoteRows) {
    $archiveBytes = [Int64]$row.archive_entry_bytes
    $localPath = Join-Path $LocalRoot ([string]$row.archive_entry)
    $localFact = Get-LocalTargetFact -Path $localPath -ArchiveBytes $archiveBytes
    [pscustomobject]@{
        audit_time = (Get-Date).ToString('yyyy-MM-ddTHH:mm:ss')
        remote_root = $RemoteRoot
        local_root = $LocalRoot
        archive = $row.archive
        archive_entry = $row.archive_entry
        archive_entry_bytes = $row.archive_entry_bytes
        archive_entry_mb = $row.archive_entry_mb
        remote_expanded_path = $row.remote_expanded_path
        remote_expanded_exists = $row.remote_expanded_exists
        remote_expanded_bytes = $row.remote_expanded_bytes
        remote_expanded_attributes = $row.remote_expanded_attributes
        remote_expanded_link_type = $row.remote_expanded_link_type
        remote_expanded_target = $row.remote_expanded_target
        remote_expanded_target_path = $row.remote_expanded_target_path
        remote_expanded_target_exists = $row.remote_expanded_target_exists
        remote_expanded_target_bytes = $row.remote_expanded_target_bytes
        remote_expanded_target_same_size = $row.remote_expanded_target_same_size
        local_expanded_path = $localPath
        local_expanded_exists = $localFact.exists
        local_expanded_bytes = $localFact.bytes
        local_expanded_attributes = $localFact.attributes
        local_expanded_link_type = $localFact.link_type
        local_expanded_target = $localFact.target
        local_expanded_target_path = $localFact.target_path
        local_expanded_target_exists = $localFact.target_exists
        local_expanded_target_bytes = $localFact.target_bytes
        local_expanded_target_same_size = $localFact.target_same_size_as_archive
        policy_signal = if ($row.remote_expanded_target_same_size -eq 'True' -or $localFact.target_same_size_as_archive -eq 'True') { 'snapshot_link_target_matches_archive_entry_size' } else { 'snapshot_link_target_not_proven_same_size' }
        note = 'Manual fixed-target symlink audit for experiments.rar CLIP cache mismatch entries; no deletion performed.'
    }
}

$rows | ConvertTo-Csv -NoTypeInformation | Set-Content -Path $LocalOut -Encoding UTF8
& ssh -p $Port -o LogLevel=ERROR $Remote "powershell -NoProfile -Command `"Remove-Item -LiteralPath '$RemoteTempWinPath' -Force -ErrorAction SilentlyContinue`"" | Out-Null
Write-Host "Wrote $LocalOut"
