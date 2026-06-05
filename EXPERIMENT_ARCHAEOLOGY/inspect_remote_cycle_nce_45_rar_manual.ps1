Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$Remote = 'administrator@100.115.18.62'
$Port = '2222'
$RemoteRoot = 'I:\Github\Latent_Style'
$ArchiveRel = 'Cycle-NCE\45.rar'
$LocalRunOut = Join-Path $PSScriptRoot 'manual_remote_cycle_nce_45_rar_run_ledger_20260605.csv'
$LocalEntryOut = Join-Path $PSScriptRoot 'manual_remote_cycle_nce_45_rar_entry_classes_20260605.csv'
$LocalTextOut = Join-Path $PSScriptRoot 'manual_remote_cycle_nce_45_rar_text_evidence_20260605.csv'

$RemoteScriptName = 'codex_inspect_cycle_nce_45_rar_manual.ps1'
$RemoteTempPath = "C:/Users/Administrator/AppData/Local/Temp/$RemoteScriptName"
$RemoteTempWinPath = "C:\Users\Administrator\AppData\Local\Temp\$RemoteScriptName"
$RemoteUnrarPath = "C:\Users\Administrator\AppData\Local\Temp\codex_UnRAR.exe"
$LocalUnrar = 'C:\Program Files\WinRAR\UnRAR.exe'

if (-not (Test-Path -LiteralPath $LocalUnrar)) {
    throw "Local UnRAR not found: $LocalUnrar"
}

$RemoteScript = @"
Set-StrictMode -Version Latest
`$ErrorActionPreference = 'Stop'
`$ProgressPreference = 'SilentlyContinue'

`$root = '$RemoteRoot'
`$archiveRel = '$ArchiveRel'
`$archivePath = Join-Path `$root `$archiveRel
`$unrar = '$RemoteUnrarPath'

function Parse-UnrarListing {
  param([string]`$ArchivePath)
  `$entries = New-Object System.Collections.Generic.List[object]
  `$raw = & `$unrar l -c- `$ArchivePath 2>&1
  `$exit = `$LASTEXITCODE
  foreach (`$line in `$raw) {
    `$text = [string]`$line
    if (`$text -match '^\s*(?<attr>\S+)\s+(?<size>\d+)\s+\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}\s+(?<name>.+)$') {
      `$name = `$Matches.name.Trim()
      `$entries.Add([pscustomobject]@{
        attr = `$Matches.attr
        size = [int64]`$Matches.size
        name = `$name
        is_dir = [bool](`$Matches.attr -match 'D')
      })
    }
  }
  return [pscustomobject]@{
    exit_code = `$exit
    entries = @(`$entries.ToArray())
    raw_head = ((`$raw | Select-Object -First 8) -join ' || ')
    raw_tail = ((`$raw | Select-Object -Last 8) -join ' || ')
  }
}

function Get-RunName {
  param([string]`$Name)
  `$norm = `$Name -replace '/', '\'
  `$parts = `$norm -split '\\'
  if (`$parts.Count -ge 3 -and `$parts[0] -eq '45') { return `$parts[1] }
  if (`$parts.Count -eq 2 -and `$parts[0] -eq '45') { return '_archive_root' }
  return ''
}

function Classify-Entry {
  param([string]`$Name)
  `$leaf = [System.IO.Path]::GetFileName((`$Name -replace '/', '\'))
  if (`$Name -match '\.(pt|pth|ckpt|safetensors)$') { return 'weight' }
  if (`$Name -match '\.(png|jpg|jpeg|webp|gif)$') { return 'generated_or_eval_image' }
  if (`$leaf -eq 'config.json') { return 'config' }
  if (`$leaf -match 'summary.*\.json$') { return 'summary_json' }
  if (`$leaf -match 'metrics.*\.csv$') { return 'metrics_csv' }
  if (`$leaf -match 'training.*\.csv$') { return 'training_csv' }
  if (`$leaf -match '\.(log|txt|md)$') { return 'log_or_note' }
  if (`$leaf -match '\.(py|yml|yaml|json|csv)$') { return 'source_or_structured' }
  return 'other'
}

function Short-Sample {
  param([object[]]`$Rows, [int]`$Limit = 12)
  if (-not `$Rows -or `$Rows.Count -eq 0) { return '' }
  return ((`$Rows | Select-Object -First `$Limit | ForEach-Object { "`$(`$_.name):`$(`$_.size)" }) -join ' | ')
}

function Read-ArchiveTextSnippet {
  param([string]`$EntryName, [int]`$MaxChars = 1800)
  `$raw = & `$unrar p -inul `$archivePath `$EntryName 2>&1
  `$exit = `$LASTEXITCODE
  `$text = (`$raw -join "`n")
  if (`$text.Length -gt `$MaxChars) {
    `$text = `$text.Substring(0, `$MaxChars)
  }
  return [pscustomobject]@{ exit_code = `$exit; snippet = `$text }
}

`$parse = Parse-UnrarListing -ArchivePath `$archivePath
`$entries = @(`$parse.entries)
`$fileEntries = @(`$entries | Where-Object { -not `$_.is_dir })
`$dirEntries = @(`$entries | Where-Object { `$_.is_dir })

`$entryRows = foreach (`$entry in `$fileEntries) {
  `$run = Get-RunName -Name `$entry.name
  `$class = Classify-Entry -Name `$entry.name
  [pscustomobject]@{
    remote_root = `$root
    archive = `$archiveRel
    run = `$run
    entry_name = `$entry.name
    bytes = [string][int64]`$entry.size
    mb = [string]([math]::Round([double]`$entry.size / 1MB, 6))
    class = `$class
  }
}

`$runRows = foreach (`$group in (`$entryRows | Group-Object run | Sort-Object Name)) {
  `$rows = @(`$group.Group)
  `$weights = @(`$rows | Where-Object { `$_.class -eq 'weight' })
  `$images = @(`$rows | Where-Object { `$_.class -eq 'generated_or_eval_image' })
  `$configs = @(`$rows | Where-Object { `$_.class -eq 'config' })
  `$summaries = @(`$rows | Where-Object { `$_.class -eq 'summary_json' })
  `$metrics = @(`$rows | Where-Object { `$_.class -eq 'metrics_csv' })
  `$logs = @(`$rows | Where-Object { `$_.class -eq 'log_or_note' -or `$_.class -eq 'training_csv' })
  [pscustomobject]@{
    remote_root = `$root
    archive = `$archiveRel
    run = `$group.Name
    file_count = [string]`$rows.Count
    total_mb = [string]([math]::Round((`$rows | ForEach-Object { [double]`$_.mb } | Measure-Object -Sum).Sum, 6))
    weight_count = [string]`$weights.Count
    weight_mb = [string]([math]::Round((`$weights | ForEach-Object { [double]`$_.mb } | Measure-Object -Sum).Sum, 6))
    image_count = [string]`$images.Count
    image_mb = [string]([math]::Round((`$images | ForEach-Object { [double]`$_.mb } | Measure-Object -Sum).Sum, 6))
    config_count = [string]`$configs.Count
    summary_count = [string]`$summaries.Count
    metrics_count = [string]`$metrics.Count
    log_or_training_count = [string]`$logs.Count
    sample_weights = ((`$weights | Select-Object -First 8 | ForEach-Object { "`$(`$_.entry_name):`$(`$_.bytes)" }) -join ' | ')
    sample_summaries = ((`$summaries | Select-Object -First 8 | ForEach-Object { `$_.entry_name }) -join ' | ')
    sample_metrics = ((`$metrics | Select-Object -First 8 | ForEach-Object { `$_.entry_name }) -join ' | ')
    sample_images = ((`$images | Select-Object -First 8 | ForEach-Object { `$_.entry_name }) -join ' | ')
    policy_signal = 'unique_archive_payload_not_expanded_currently'
  }
}

`$textCandidates = @(`$fileEntries | Where-Object {
  `$_.name -match '\\config\.json$' -or
  `$_.name -match '\\summary\.json$' -or
  `$_.name -match '\\run_summary\.json$' -or
  `$_.name -match '\\metrics\.csv$'
} | Sort-Object name)

`$textRows = foreach (`$entry in `$textCandidates) {
  `$run = Get-RunName -Name `$entry.name
  `$snippet = Read-ArchiveTextSnippet -EntryName `$entry.name
  [pscustomobject]@{
    remote_root = `$root
    archive = `$archiveRel
    run = `$run
    entry_name = `$entry.name
    bytes = [string][int64]`$entry.size
    class = Classify-Entry -Name `$entry.name
    unrar_p_exit_code = [string]`$snippet.exit_code
    snippet = `$snippet.snippet
  }
}

[pscustomobject]@{
  archive_summary = @([pscustomobject]@{
    remote_root = `$root
    archive = `$archiveRel
    unrar_exit_code = [string]`$parse.exit_code
    entry_count = [string]`$entries.Count
    file_entry_count = [string]`$fileEntries.Count
    directory_entry_count = [string]`$dirEntries.Count
    raw_head = `$parse.raw_head
    raw_tail = `$parse.raw_tail
  })
  run_rows = @(`$runRows)
  entry_rows = @(`$entryRows)
  text_rows = @(`$textRows)
} | ConvertTo-Json -Depth 8
"@

& scp -P $Port -o LogLevel=ERROR $LocalUnrar "${Remote}:C:/Users/Administrator/AppData/Local/Temp/codex_UnRAR.exe" | Out-Null
$LocalRemoteScript = Join-Path $env:TEMP $RemoteScriptName
$RemoteScript | Set-Content -Path $LocalRemoteScript -Encoding UTF8
& scp -P $Port -o LogLevel=ERROR $LocalRemoteScript "${Remote}:$RemoteTempPath" | Out-Null

$raw = & ssh -p $Port -o LogLevel=ERROR $Remote "powershell -NoProfile -ExecutionPolicy Bypass -File `"$RemoteTempWinPath`""
$json = ($raw -join "`n")
$obj = $json | ConvertFrom-Json

@($obj.run_rows) | ConvertTo-Csv -NoTypeInformation | Set-Content -Path $LocalRunOut -Encoding UTF8
@($obj.entry_rows) | ConvertTo-Csv -NoTypeInformation | Set-Content -Path $LocalEntryOut -Encoding UTF8
@($obj.text_rows) | ConvertTo-Csv -NoTypeInformation | Set-Content -Path $LocalTextOut -Encoding UTF8

& ssh -p $Port -o LogLevel=ERROR $Remote "powershell -NoProfile -Command `"Remove-Item -LiteralPath '$RemoteTempWinPath' -Force -ErrorAction SilentlyContinue; Remove-Item -LiteralPath '$RemoteUnrarPath' -Force -ErrorAction SilentlyContinue`"" | Out-Null
Write-Host "Wrote $LocalRunOut"
Write-Host "Wrote $LocalEntryOut"
Write-Host "Wrote $LocalTextOut"
