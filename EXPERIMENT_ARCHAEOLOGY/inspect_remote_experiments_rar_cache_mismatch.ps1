Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$Remote = 'administrator@100.115.18.62'
$Port = '2222'
$RemoteRoot = 'I:\Github\Latent_Style'
$LocalRoot = 'G:\GitHub\Latent_Style'
$LocalOut = Join-Path $PSScriptRoot 'manual_remote_experiments_rar_cache_mismatch_20260605.csv'

$RemoteScriptName = 'codex_inspect_experiments_rar_cache_mismatch_light.ps1'
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

function Get-LocalFact {
    param(
        [string]$Label,
        [string]$Path,
        [Int64]$ArchiveBytes
    )
    if (Test-Path -LiteralPath $Path) {
        $item = Get-Item -LiteralPath $Path -Force
        return [pscustomobject]@{
            label = $Label
            path = $Path
            exists = 'True'
            bytes = [string][Int64]$item.Length
            same_size_as_archive = [string]([Int64]$item.Length -eq $ArchiveBytes)
            attributes = [string]$item.Attributes
            link_type = [string]$item.LinkType
            target = if ($null -ne $item.Target) { [string]($item.Target -join ';') } else { '' }
            last_write = $item.LastWriteTime.ToString('yyyy-MM-dd HH:mm:ss')
        }
    }
    return [pscustomobject]@{
        label = $Label
        path = $Path
        exists = 'False'
        bytes = ''
        same_size_as_archive = 'False'
        attributes = ''
        link_type = ''
        target = ''
        last_write = ''
    }
}

function Join-Facts {
    param([object[]]$Facts)
    if (-not $Facts -or $Facts.Count -eq 0) { return '' }
    return (($Facts | ForEach-Object { "$($_.label):exists=$($_.exists):bytes=$($_.bytes):same_size=$($_.same_size_as_archive):attrs=$($_.attributes):link=$($_.link_type):target=$($_.target):path=$($_.path)" }) -join ' | ')
}

$RemoteEntriesLiteral = ($entries | ConvertTo-Json -Depth 3 -Compress)
$RemoteScript = @"
Set-StrictMode -Version Latest
`$ErrorActionPreference = 'Stop'
`$root = '$RemoteRoot'
`$entries = '$RemoteEntriesLiteral' | ConvertFrom-Json

function Get-Fact {
  param([string]`$Label, [string]`$Path, [Int64]`$ArchiveBytes)
  if (Test-Path -LiteralPath `$Path) {
    `$item = Get-Item -LiteralPath `$Path -Force
    return [pscustomobject]@{
      label = `$Label
      path = `$Path
      exists = 'True'
      bytes = [string][Int64]`$item.Length
      same_size_as_archive = [string]([Int64]`$item.Length -eq `$ArchiveBytes)
      attributes = [string]`$item.Attributes
      link_type = [string]`$item.LinkType
      target = if (`$null -ne `$item.Target) { [string](`$item.Target -join ';') } else { '' }
      last_write = `$item.LastWriteTime.ToString('yyyy-MM-dd HH:mm:ss')
    }
  }
  return [pscustomobject]@{
    label = `$Label
    path = `$Path
    exists = 'False'
    bytes = ''
    same_size_as_archive = 'False'
    attributes = ''
    link_type = ''
    target = ''
    last_write = ''
  }
}

function Join-Facts {
  param([object[]]`$Facts)
  if (-not `$Facts -or `$Facts.Count -eq 0) { return '' }
  return ((`$Facts | ForEach-Object { "`$(`$_.label):exists=`$(`$_.exists):bytes=`$(`$_.bytes):same_size=`$(`$_.same_size_as_archive):attrs=`$(`$_.attributes):link=`$(`$_.link_type):target=`$(`$_.target):path=`$(`$_.path)" }) -join ' | ')
}

`$out = foreach (`$e in `$entries) {
  `$entry = [string]`$e.entry
  `$archiveBytes = [Int64]`$e.archive_bytes
  `$stripped = `$entry -replace '^experiments\\', ''
  `$facts = @(
    (Get-Fact -Label 'remote_expanded_experiments' -Path (Join-Path `$root `$entry) -ArchiveBytes `$archiveBytes),
    (Get-Fact -Label 'remote_root_eval_cache' -Path (Join-Path `$root `$stripped) -ArchiveBytes `$archiveBytes),
    (Get-Fact -Label 'remote_schrodingerbridge_eval_cache' -Path (Join-Path `$root (Join-Path 'SchrodingerBridge' `$stripped)) -ArchiveBytes `$archiveBytes)
  )
  `$sameSizeCount = @(`$facts | Where-Object { `$_.same_size_as_archive -eq 'True' }).Count
  `$expanded = `$facts | Where-Object { `$_.label -eq 'remote_expanded_experiments' } | Select-Object -First 1
  [pscustomobject]@{
    archive = 'experiments.rar'
    archive_entry = `$entry
    stripped_cache_path = `$stripped
    archive_entry_bytes = [string]`$archiveBytes
    archive_entry_mb = [string]([math]::Round([double]`$archiveBytes / 1MB, 6))
    remote_expanded_exists = `$expanded.exists
    remote_expanded_bytes = `$expanded.bytes
    remote_expanded_same_size = `$expanded.same_size_as_archive
    remote_expanded_attributes = `$expanded.attributes
    remote_expanded_link_type = `$expanded.link_type
    remote_expanded_target = `$expanded.target
    remote_same_size_candidate_count = [string]`$sameSizeCount
    remote_fact_summary = Join-Facts `$facts
    remote_policy_signal = if (`$sameSizeCount -gt 0) { 'archive_entry_has_remote_same_size_candidate' } else { 'archive_entry_has_no_remote_same_size_candidate' }
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
    throw "Remote experiments.rar light mismatch audit did not return CSV. Output: $joined"
}

$remoteRows = $csvLines | ConvertFrom-Csv
$rows = foreach ($row in $remoteRows) {
    $archiveBytes = [Int64]$row.archive_entry_bytes
    $stripped = [string]$row.stripped_cache_path
    $localFacts = @(
        (Get-LocalFact -Label 'local_root_eval_cache' -Path (Join-Path $LocalRoot $stripped) -ArchiveBytes $archiveBytes),
        (Get-LocalFact -Label 'local_schrodingerbridge_eval_cache' -Path (Join-Path $LocalRoot (Join-Path 'SchrodingerBridge' $stripped)) -ArchiveBytes $archiveBytes),
        (Get-LocalFact -Label 'local_experiments_cache' -Path (Join-Path $LocalRoot ([string]$row.archive_entry)) -ArchiveBytes $archiveBytes)
    )
    $localSameSizeCount = @($localFacts | Where-Object { $_.same_size_as_archive -eq 'True' }).Count
    [pscustomobject]@{
        audit_time = (Get-Date).ToString('yyyy-MM-ddTHH:mm:ss')
        remote_root = $RemoteRoot
        local_root = $LocalRoot
        archive = $row.archive
        archive_entry = $row.archive_entry
        stripped_cache_path = $row.stripped_cache_path
        archive_entry_bytes = $row.archive_entry_bytes
        archive_entry_mb = $row.archive_entry_mb
        remote_expanded_exists = $row.remote_expanded_exists
        remote_expanded_bytes = $row.remote_expanded_bytes
        remote_expanded_same_size = $row.remote_expanded_same_size
        remote_expanded_attributes = $row.remote_expanded_attributes
        remote_expanded_link_type = $row.remote_expanded_link_type
        remote_expanded_target = $row.remote_expanded_target
        remote_same_size_candidate_count = $row.remote_same_size_candidate_count
        remote_fact_summary = $row.remote_fact_summary
        local_same_size_candidate_count = [string]$localSameSizeCount
        local_fact_summary = Join-Facts $localFacts
        policy_signal = if (([int]$row.remote_same_size_candidate_count + [int]$localSameSizeCount) -gt 0) { 'archive_entry_has_same_size_candidate_somewhere' } else { 'archive_entry_has_no_same_size_candidate_in_checked_roots' }
        note = 'Fixed-target manual audit of the 9 known experiments.rar CLIP cache mismatch entries; no deletion performed.'
    }
}

$rows | ConvertTo-Csv -NoTypeInformation | Set-Content -Path $LocalOut -Encoding UTF8
& ssh -p $Port -o LogLevel=ERROR $Remote "powershell -NoProfile -Command `"Remove-Item -LiteralPath '$RemoteTempWinPath' -Force -ErrorAction SilentlyContinue`"" | Out-Null
Write-Host "Wrote $LocalOut"
