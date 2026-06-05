Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$Remote = 'administrator@100.115.18.62'
$Port = '2222'
$RemoteRoot = 'I:\Github\Latent_Style'
$ArchiveRel = 'experiments.rar'
$CleanupOut = Join-Path $PSScriptRoot 'cleanup\manual_remote_experiments_rar_resolved_duplicate_cleanup_20260605.csv'
$VerifyOut = Join-Path $PSScriptRoot 'manual_remote_experiments_rar_resolved_duplicate_post_delete_verify_20260605.csv'

$RemoteScriptName = 'codex_delete_experiments_rar_resolved_duplicate.ps1'
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

$RemoteEntriesLiteral = ($entries | ConvertTo-Json -Depth 3 -Compress)
$RemoteScript = @"
Set-StrictMode -Version Latest
`$ErrorActionPreference = 'Stop'
`$root = '$RemoteRoot'
`$archiveRel = '$ArchiveRel'
`$entries = '$RemoteEntriesLiteral' | ConvertFrom-Json
`$run = (Get-Date).ToString('yyyy-MM-ddTHH:mm:ss')

function Resolve-WithinRoot {
  param([string]`$Relative)
  `$path = Join-Path `$root `$Relative
  `$full = [System.IO.Path]::GetFullPath(`$path)
  `$rootFull = [System.IO.Path]::GetFullPath(`$root)
  if (-not `$full.StartsWith(`$rootFull, [System.StringComparison]::OrdinalIgnoreCase)) {
    throw "Resolved path escapes root: `$Relative -> `$full"
  }
  return `$full
}

function Get-LinkTargetFact {
  param([string]`$Path, [Int64]`$ArchiveBytes)
  if (-not (Test-Path -LiteralPath `$Path)) {
    return [pscustomobject]@{ exists='False'; target_path=''; target_exists='False'; target_bytes=''; target_same_size='False'; link_type=''; attributes='' }
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
    target_path = `$targetPath
    target_exists = `$targetExists
    target_bytes = `$targetBytes
    target_same_size = `$targetSame
    link_type = [string]`$item.LinkType
    attributes = [string]`$item.Attributes
  }
}

`$archivePath = Resolve-WithinRoot -Relative `$archiveRel
`$existsBefore = Test-Path -LiteralPath `$archivePath
if (-not `$existsBefore) {
  throw "Archive does not exist before deletion: `$archivePath"
}
`$item = Get-Item -LiteralPath `$archivePath -Force
`$beforeBytes = [Int64]`$item.Length
`$beforeMb = [math]::Round([double]`$beforeBytes / 1MB, 6)
`$lastWrite = `$item.LastWriteTime.ToString('yyyy-MM-dd HH:mm:ss')

Remove-Item -LiteralPath `$archivePath -Force
`$postExists = Test-Path -LiteralPath `$archivePath

`$cleanup = [pscustomobject]@{
  cleanup_run = `$run
  remote_root = `$root
  relative_path = `$archiveRel
  exists_before = [string]`$existsBefore
  before_bytes = [string]`$beforeBytes
  before_mb = [string]`$beforeMb
  last_write_time = `$lastWrite
  status = if (`$postExists) { 'delete_failed_still_exists' } else { 'deleted' }
  post_exists = [string]`$postExists
  reason = 'Resolved duplicate: all original mismatches are HF snapshot symlinks with same-size blob targets.'
}

`$verify = New-Object System.Collections.Generic.List[object]
`$verify.Add([pscustomobject]@{
  verify_run = `$run
  remote_root = `$root
  relative_path = `$archiveRel
  expected = 'absent'
  exists = [string](Test-Path -LiteralPath `$archivePath)
  pass = [string](-not (Test-Path -LiteralPath `$archivePath))
  reason = 'deleted resolved duplicate archive'
})
`$experimentsPath = Resolve-WithinRoot -Relative 'experiments'
`$verify.Add([pscustomobject]@{
  verify_run = `$run
  remote_root = `$root
  relative_path = 'experiments'
  expected = 'present'
  exists = [string](Test-Path -LiteralPath `$experimentsPath)
  pass = [string](Test-Path -LiteralPath `$experimentsPath)
  reason = 'expanded experiments evidence retained'
})

foreach (`$e in `$entries) {
  `$entry = [string]`$e.entry
  `$archiveBytes = [Int64]`$e.archive_bytes
  `$path = Resolve-WithinRoot -Relative `$entry
  `$fact = Get-LinkTargetFact -Path `$path -ArchiveBytes `$archiveBytes
  `$verify.Add([pscustomobject]@{
    verify_run = `$run
    remote_root = `$root
    relative_path = `$entry
    expected = 'symlink target same-size'
    exists = `$fact.exists
    pass = [string](`$fact.exists -eq 'True' -and `$fact.target_exists -eq 'True' -and `$fact.target_same_size -eq 'True')
    reason = "link_type=`$(`$fact.link_type); attrs=`$(`$fact.attributes); target=`$(`$fact.target_path); target_exists=`$(`$fact.target_exists); target_bytes=`$(`$fact.target_bytes); archive_bytes=`$archiveBytes"
  })
}

[pscustomobject]@{
  cleanup = @(`$cleanup)
  verify = @(`$verify.ToArray())
} | ConvertTo-Json -Depth 6
"@

$LocalRemoteScript = Join-Path $env:TEMP $RemoteScriptName
$RemoteScript | Set-Content -Path $LocalRemoteScript -Encoding UTF8
& scp -P $Port -o LogLevel=ERROR $LocalRemoteScript "${Remote}:$RemoteTempPath" | Out-Null
$raw = & ssh -p $Port -o LogLevel=ERROR $Remote "powershell -NoProfile -ExecutionPolicy Bypass -File `"$RemoteTempWinPath`""
$json = ($raw -join "`n")
$obj = $json | ConvertFrom-Json

@($obj.cleanup) | ConvertTo-Csv -NoTypeInformation | Set-Content -Path $CleanupOut -Encoding UTF8
@($obj.verify) | ConvertTo-Csv -NoTypeInformation | Set-Content -Path $VerifyOut -Encoding UTF8

& ssh -p $Port -o LogLevel=ERROR $Remote "powershell -NoProfile -Command `"Remove-Item -LiteralPath '$RemoteTempWinPath' -Force -ErrorAction SilentlyContinue`"" | Out-Null
Write-Host "Wrote $CleanupOut"
Write-Host "Wrote $VerifyOut"
