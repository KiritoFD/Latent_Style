Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$Remote = 'administrator@100.115.18.62'
$Port = '2222'
$RemoteRoot = 'I:\Github\Latent_Style'
$LocalOut = Join-Path $PSScriptRoot 'manual_remote_rar_provenance_deep_20260605.csv'

$RemoteScriptName = 'codex_inspect_remote_rar_provenance_deep.ps1'
$RemoteTempPath = "C:/Users/Administrator/AppData/Local/Temp/$RemoteScriptName"
$RemoteTempWinPath = "C:\Users\Administrator\AppData\Local\Temp\$RemoteScriptName"
$RemoteUnrarPath = "C:\Users\Administrator\AppData\Local\Temp\codex_UnRAR.exe"

$LocalUnrar = 'C:\Program Files\WinRAR\UnRAR.exe'
if (-not (Test-Path -LiteralPath $LocalUnrar)) {
    throw "Local UnRAR not found: $LocalUnrar"
}

$RemoteScript = @'
Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
$ProgressPreference = 'SilentlyContinue'

$root = 'I:\Github\Latent_Style'
$unrar = 'C:\Users\Administrator\AppData\Local\Temp\codex_UnRAR.exe'

$archives = @(
  [pscustomobject]@{
    group = 'experiments.rar'
    list_archive = 'experiments.rar'
    parts = 'experiments.rar'
    compare_roots = @('experiments')
  },
  [pscustomobject]@{
    group = 'Cycle-NCE/Gate.rar'
    list_archive = 'Cycle-NCE\Gate.rar'
    parts = 'Cycle-NCE\Gate.rar'
    compare_roots = @('Cycle-NCE')
  },
  [pscustomobject]@{
    group = 'Cycle-NCE/Attn_48.part*.rar'
    list_archive = 'Cycle-NCE\Attn_48.part1.rar'
    parts = 'Cycle-NCE\Attn_48.part1.rar;Cycle-NCE\Attn_48.part2.rar;Cycle-NCE\Attn_48.part3.rar'
    compare_roots = @('Cycle-NCE')
  },
  [pscustomobject]@{
    group = 'Cycle-NCE/chess.part*.rar'
    list_archive = 'Cycle-NCE\chess.part1.rar'
    parts = 'Cycle-NCE\chess.part1.rar;Cycle-NCE\chess.part2.rar'
    compare_roots = @('Cycle-NCE')
  },
  [pscustomobject]@{
    group = 'Cycle-NCE/45.rar'
    list_archive = 'Cycle-NCE\45.rar'
    parts = 'Cycle-NCE\45.rar'
    compare_roots = @('Cycle-NCE')
  }
)

function Get-PartInfo {
  param([string]$Parts)
  $rows = @()
  foreach ($rel in ($Parts -split ';')) {
    $path = Join-Path $root $rel
    if (Test-Path -LiteralPath $path) {
      $item = Get-Item -LiteralPath $path -Force
      $rows += [pscustomobject]@{
        rel = $rel
        mb = [math]::Round([double]$item.Length / 1MB, 6)
        last_write = $item.LastWriteTime
        exists = $true
      }
    } else {
      $rows += [pscustomobject]@{ rel = $rel; mb = 0; last_write = ''; exists = $false }
    }
  }
  return $rows
}

function Parse-UnrarListing {
  param([string]$ArchivePath)
  $entries = New-Object System.Collections.Generic.List[object]
  $raw = & $unrar l -c- $ArchivePath 2>&1
  $exit = $LASTEXITCODE
  foreach ($line in $raw) {
    $text = [string]$line
    if ($text -match '^\s*(?<attr>\S+)\s+(?<size>\d+)\s+\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}\s+(?<name>.+)$') {
      $entries.Add([pscustomobject]@{
        attr = $Matches.attr
        size = [int64]$Matches.size
        name = $Matches.name.Trim()
        is_dir = [bool]($Matches.attr -match 'D')
      })
    }
  }
  return ,[pscustomobject]@{
    entries = @($entries.ToArray())
    exit_code = $exit
    raw_head = (($raw | Select-Object -First 8) -join ' || ')
    raw_tail = (($raw | Select-Object -Last 8) -join ' || ')
  }
}

function Build-SizeMap {
  param([string[]]$CompareRoots)
  $map = @{}
  foreach ($base in $CompareRoots) {
    $basePath = if ([string]::IsNullOrWhiteSpace($base)) { $root } else { Join-Path $root $base }
    if (-not (Test-Path -LiteralPath $basePath)) { continue }
    $files = @(Get-ChildItem -LiteralPath $basePath -Recurse -File -Force -ErrorAction SilentlyContinue)
    foreach ($file in $files) {
      $relRoot = $file.FullName.Substring($root.Length + 1).ToLowerInvariant()
      $map[$relRoot] = [int64]$file.Length
      if ($basePath.Length -lt $file.FullName.Length) {
        $relBase = $file.FullName.Substring($basePath.Length + 1).ToLowerInvariant()
        $map[$relBase] = [int64]$file.Length
      }
    }
  }
  return ,$map
}

function Find-SameSizeExistingInMap {
  param(
    [hashtable]$SizeMap,
    [string]$EntryName,
    [int64]$EntrySize,
    [string]$ArchiveRel
  )
  $normalized = ($EntryName -replace '/', '\').ToLowerInvariant()
  $archiveParentRel = Split-Path -Path $ArchiveRel -Parent
  $candidateKeys = New-Object System.Collections.Generic.List[string]
  $candidateKeys.Add($normalized)
  if (-not [string]::IsNullOrWhiteSpace($archiveParentRel)) {
    $candidateKeys.Add(((Join-Path $archiveParentRel $normalized).ToLowerInvariant()))
  }

  foreach ($key in ($candidateKeys | Select-Object -Unique)) {
    if ($SizeMap.ContainsKey($key)) {
      $actual = [int64]$SizeMap[$key]
      if ($actual -eq $EntrySize) {
        return [pscustomobject]@{ exists = $true; rel = $key; size_match = $true; actual_size = $actual }
      }
      return [pscustomobject]@{ exists = $true; rel = $key; size_match = $false; actual_size = $actual }
    }
  }
  return [pscustomobject]@{ exists = $false; rel = ''; size_match = $false; actual_size = '' }
}

function Short-Sample {
  param([object[]]$Rows, [int]$Limit = 20)
  if (-not $Rows -or $Rows.Count -eq 0) { return '' }
  return (($Rows | Select-Object -First $Limit | ForEach-Object { "$($_.name):$($_.size)" }) -join ' | ')
}

$out = foreach ($archive in $archives) {
  $archivePath = Join-Path $root $archive.list_archive
  $partInfo = @(Get-PartInfo -Parts $archive.parts)
  $partMb = ($partInfo | Measure-Object -Property mb -Sum).Sum
  if ($null -eq $partMb) { $partMb = 0 }

  $parse = Parse-UnrarListing -ArchivePath $archivePath
  $entries = @($parse.entries)
  $fileEntries = @($entries | Where-Object { -not $_.is_dir })
  $dirEntries = @($entries | Where-Object { $_.is_dir })
  $sizeMap = Build-SizeMap -CompareRoots $archive.compare_roots
  $same = 0
  $missing = New-Object System.Collections.Generic.List[object]
  $mismatch = New-Object System.Collections.Generic.List[object]
  $weightMissing = 0
  $weightMismatch = 0
  $nonWeightMissing = 0
  $nonWeightMismatch = 0

  foreach ($entry in $fileEntries) {
    $check = Find-SameSizeExistingInMap -SizeMap $sizeMap -EntryName $entry.name -EntrySize $entry.size -ArchiveRel $archive.list_archive
    $isWeight = [bool]($entry.name -match '\.(pt|pth|ckpt|safetensors)$')
    if ($check.exists -and $check.size_match) {
      $same++
    } elseif ($check.exists) {
      $mismatch.Add($entry)
      if ($isWeight) { $weightMismatch++ } else { $nonWeightMismatch++ }
    } else {
      $missing.Add($entry)
      if ($isWeight) { $weightMissing++ } else { $nonWeightMissing++ }
    }
  }

  $decision = 'retain_pending_review'
  if ($parse.exit_code -ne 0 -and $fileEntries.Count -eq 0) {
    $decision = 'retain_unrar_failed'
  } elseif ($fileEntries.Count -gt 0 -and $missing.Count -eq 0 -and $mismatch.Count -eq 0) {
    $decision = 'delete_candidate_all_entries_same_size_existing'
  } elseif ($fileEntries.Count -gt 0 -and $nonWeightMissing -eq 0 -and $nonWeightMismatch -eq 0 -and ($weightMissing + $weightMismatch) -gt 0) {
    $decision = 'review_delete_candidate_only_weight_entries_not_redundant'
  }

  [pscustomobject]@{
    remote_root = $root
    archive_group = $archive.group
    list_archive = $archive.list_archive
    parts = $archive.parts
    parts_exist = (($partInfo | ForEach-Object { "$($_.rel)=$($_.exists)" }) -join ';')
    parts_total_mb = [math]::Round([double]$partMb, 6)
    unrar_exit_code = $parse.exit_code
    entry_count = $entries.Count
    file_entry_count = $fileEntries.Count
    directory_entry_count = $dirEntries.Count
    expanded_size_map_entries = $sizeMap.Count
    same_size_existing_count = $same
    missing_count = $missing.Count
    mismatch_count = $mismatch.Count
    weight_missing_count = $weightMissing
    weight_mismatch_count = $weightMismatch
    nonweight_missing_count = $nonWeightMissing
    nonweight_mismatch_count = $nonWeightMismatch
    sample_missing = Short-Sample @($missing.ToArray()) 20
    sample_mismatch = Short-Sample @($mismatch.ToArray()) 20
    raw_head = $parse.raw_head
    raw_tail = $parse.raw_tail
    decision = $decision
  }
}

$out | ConvertTo-Csv -NoTypeInformation
'@

& scp -P $Port -o LogLevel=ERROR $LocalUnrar "${Remote}:C:/Users/Administrator/AppData/Local/Temp/codex_UnRAR.exe" | Out-Null
$LocalRemoteScript = Join-Path $env:TEMP $RemoteScriptName
$RemoteScript | Set-Content -Path $LocalRemoteScript -Encoding UTF8
& scp -P $Port -o LogLevel=ERROR $LocalRemoteScript "${Remote}:$RemoteTempPath" | Out-Null

$raw = & ssh -p $Port -o LogLevel=ERROR $Remote "powershell -NoProfile -ExecutionPolicy Bypass -File `"$RemoteTempWinPath`""
$csvLines = @($raw | Where-Object { $_ -match '^"' })
if (-not $csvLines -or $csvLines.Count -lt 2) {
    $joined = ($raw -join "`n")
    throw "Remote RAR provenance did not return CSV. Output: $joined"
}
$csvLines | Set-Content -Path $LocalOut -Encoding UTF8
Write-Host "Wrote $LocalOut"
