$ErrorActionPreference = 'Stop'
$ProgressPreference = 'SilentlyContinue'

Add-Type -AssemblyName System.IO.Compression.FileSystem

$root = 'I:\Github\Latent_Style'
$rows = New-Object System.Collections.Generic.List[object]

function New-Row {
    param(
        [string]$RowType,
        [string]$RelativePath,
        [string]$ComparePath = '',
        [string]$Exists = '',
        [string]$Bytes = '',
        [string]$MB = '',
        [string]$Sha256 = '',
        [string]$LastWrite = '',
        [string]$FileCount = '',
        [string]$TotalMB = '',
        [string]$EntryCount = '',
        [string]$TotalUncompressedMB = '',
        [string]$ExistingEntries = '',
        [string]$SameSizeEntries = '',
        [string]$MissingEntries = '',
        [string]$MismatchedSizeEntries = '',
        [string]$TopDirs = '',
        [string]$Note = ''
    )

    [pscustomobject]@{
        row_type = $RowType
        relative_path = $RelativePath
        compare_path = $ComparePath
        exists = $Exists
        bytes = $Bytes
        mb = $MB
        sha256 = $Sha256
        last_write = $LastWrite
        file_count = $FileCount
        total_mb = $TotalMB
        entry_count = $EntryCount
        total_uncompressed_mb = $TotalUncompressedMB
        existing_entries = $ExistingEntries
        same_size_entries = $SameSizeEntries
        missing_entries = $MissingEntries
        mismatched_size_entries = $MismatchedSizeEntries
        top_dirs = $TopDirs
        note = $Note
    }
}

foreach ($cmd in @('7z.exe', '7za.exe', 'rar.exe', 'unrar.exe', 'WinRAR.exe')) {
    $found = @(Get-Command $cmd -ErrorAction SilentlyContinue | ForEach-Object { $_.Source })
    $rows.Add((New-Row -RowType 'tool' -RelativePath $cmd -Exists ([string]($found.Count -gt 0)) -Note (($found -join ';'))))
}

$archiveRels = @(
    'eval_cache.zip',
    'experiments.rar',
    'Cycle-NCE\Gate.rar',
    'Cycle-NCE\1-decoder-patch5-15_eAzEC.zip',
    'Cycle-NCE\Attn_48.part1.rar',
    'Cycle-NCE\Attn_48.part2.rar',
    'Cycle-NCE\Attn_48.part3.rar',
    'Cycle-NCE\chess.part1.rar',
    'Cycle-NCE\chess.part2.rar',
    'Cycle-NCE\45.rar',
    'Cycle-NCE\src\45.rar',
    'Cycle-NCE\src_BGmRM.7z',
    'Cycle-NCE\summary_fhJh7.zip'
)

foreach ($rel in $archiveRels) {
    $path = Join-Path $root $rel
    if (Test-Path -LiteralPath $path) {
        $item = Get-Item -LiteralPath $path
        $rows.Add((New-Row -RowType 'archive_file' -RelativePath $rel -Exists 'True' -Bytes ([string]$item.Length) -MB ([string][math]::Round($item.Length / 1MB, 3)) -LastWrite ([string]$item.LastWriteTime)))
    } else {
        $rows.Add((New-Row -RowType 'archive_file' -RelativePath $rel -Exists 'False'))
    }
}

$dirRels = @(
    'eval_cache',
    'experiments',
    'Cycle-NCE\Gate',
    'Cycle-NCE\1-decoder-patch5-15',
    'Cycle-NCE\Attn_48',
    'Cycle-NCE\chess',
    'Cycle-NCE\45',
    'Cycle-NCE\src'
)

foreach ($rel in $dirRels) {
    $path = Join-Path $root $rel
    if (Test-Path -LiteralPath $path) {
        $files = @(Get-ChildItem -LiteralPath $path -Recurse -File -Force -ErrorAction SilentlyContinue)
        $immediateDirs = (Get-ChildItem -LiteralPath $path -Force -Directory -ErrorAction SilentlyContinue | Sort-Object Name | Select-Object -First 20 | ForEach-Object { $_.Name }) -join ';'
        $immediateFiles = (Get-ChildItem -LiteralPath $path -Force -File -ErrorAction SilentlyContinue | Sort-Object Name | Select-Object -First 20 | ForEach-Object { $_.Name + ':' + [math]::Round($_.Length / 1MB, 3) + 'MB' }) -join ';'
        $rows.Add((New-Row -RowType 'matching_dir' -RelativePath $rel -Exists 'True' -FileCount ([string]$files.Count) -TotalMB ([string][math]::Round((($files | Measure-Object Length -Sum).Sum) / 1MB, 3)) -LastWrite ([string](Get-Item -LiteralPath $path).LastWriteTime) -TopDirs $immediateDirs -Note $immediateFiles))
    } else {
        $rows.Add((New-Row -RowType 'matching_dir' -RelativePath $rel -Exists 'False'))
    }
}

$hashRels = @(
    'Cycle-NCE\45.rar',
    'Cycle-NCE\src\45.rar',
    'eval_cache.zip',
    'Cycle-NCE\summary_fhJh7.zip',
    'Cycle-NCE\src_BGmRM.7z'
)

foreach ($rel in $hashRels) {
    $path = Join-Path $root $rel
    if (Test-Path -LiteralPath $path) {
        $item = Get-Item -LiteralPath $path
        $hash = Get-FileHash -LiteralPath $path -Algorithm SHA256
        $rows.Add((New-Row -RowType 'sha256' -RelativePath $rel -Exists 'True' -Bytes ([string]$item.Length) -MB ([string][math]::Round($item.Length / 1MB, 3)) -Sha256 $hash.Hash -LastWrite ([string]$item.LastWriteTime)))
    }
}

function Add-ZipCompare {
    param(
        [Parameter(Mandatory = $true)][string]$ZipRel,
        [Parameter(Mandatory = $true)][string[]]$CompareRoots
    )

    $zipPath = Join-Path $root $ZipRel
    if (!(Test-Path -LiteralPath $zipPath)) {
        return
    }

    $zip = [System.IO.Compression.ZipFile]::OpenRead($zipPath)
    try {
        $entries = @($zip.Entries | Where-Object { $_.Name -ne '' })
        $entryRows = foreach ($entry in $entries) {
            $entryName = $entry.FullName -replace '/', '\'
            $exists = $false
            $sameSize = $false
            $matchPath = ''

            foreach ($compareRoot in $CompareRoots) {
                $candidate = Join-Path $compareRoot $entryName
                if (Test-Path -LiteralPath $candidate) {
                    $item = Get-Item -LiteralPath $candidate
                    $exists = $true
                    $sameSize = ($item.Length -eq $entry.Length)
                    $matchPath = $candidate.Substring($root.Length + 1)
                    break
                }
            }

            [pscustomobject]@{
                entry = $entryName
                bytes = $entry.Length
                exists = $exists
                same_size = $sameSize
                match_path = $matchPath
            }
        }

        $missingOrMismatch = @($entryRows | Where-Object { !$_.exists -or ($_.exists -and !$_.same_size) } | Select-Object -First 30 | ForEach-Object {
            $_.entry + '|exists=' + $_.exists + '|same=' + $_.same_size + '|bytes=' + $_.bytes + '|match=' + $_.match_path
        })
        $sample = @($entryRows | Select-Object -First 20 | ForEach-Object {
            $_.entry + '|exists=' + $_.exists + '|same=' + $_.same_size + '|bytes=' + $_.bytes
        })

        $topDirs = ($entries | ForEach-Object { ($_.FullName -split '/')[0] } | Group-Object | Sort-Object Count -Descending | Select-Object -First 10 | ForEach-Object { $_.Name + ':' + $_.Count }) -join ';'
        $rows.Add((New-Row -RowType 'zip_compare_summary' -RelativePath $ZipRel -Exists 'True' -EntryCount ([string]$entries.Count) -TotalUncompressedMB ([string][math]::Round((($entries | Measure-Object Length -Sum).Sum) / 1MB, 3)) -ExistingEntries ([string]@($entryRows | Where-Object exists).Count) -SameSizeEntries ([string]@($entryRows | Where-Object same_size).Count) -MissingEntries ([string]@($entryRows | Where-Object { !$_.exists }).Count) -MismatchedSizeEntries ([string]@($entryRows | Where-Object { $_.exists -and !$_.same_size }).Count) -TopDirs $topDirs -Note ('missing_or_mismatch=' + ($missingOrMismatch -join ';') + ' sample=' + ($sample -join ';'))))
    } finally {
        $zip.Dispose()
    }
}

Add-ZipCompare -ZipRel 'eval_cache.zip' -CompareRoots @($root, (Join-Path $root 'eval_cache'))
Add-ZipCompare -ZipRel 'Cycle-NCE\1-decoder-patch5-15_eAzEC.zip' -CompareRoots @((Join-Path $root 'Cycle-NCE'), (Join-Path $root 'Cycle-NCE\1-decoder-patch5-15'), (Join-Path $root 'experiments'))
Add-ZipCompare -ZipRel 'Cycle-NCE\summary_fhJh7.zip' -CompareRoots @((Join-Path $root 'Cycle-NCE'))

$rows | ConvertTo-Csv -NoTypeInformation
