Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$Remote = 'administrator@100.115.18.62'
$Port = '2222'
$RemoteRoot = 'I:\Github\Latent_Style'
$ArchiveRel = 'Cycle-NCE\45.rar'
$LocalOut = Join-Path $PSScriptRoot 'manual_remote_cycle_nce_45_rar_summary_metrics_20260605.csv'

$RemoteScriptName = 'codex_extract_cycle_nce_45_rar_summary_metrics.ps1'
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
  foreach (`$line in `$raw) {
    `$text = [string]`$line
    if (`$text -match '^\s*(?<attr>\S+)\s+(?<size>\d+)\s+\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}\s+(?<name>.+)$') {
      `$entries.Add([pscustomobject]@{
        attr = `$Matches.attr
        size = [int64]`$Matches.size
        name = `$Matches.name.Trim()
        is_dir = [bool](`$Matches.attr -match 'D')
      })
    }
  }
  return @(`$entries.ToArray())
}

function Get-RunName {
  param([string]`$Name)
  `$norm = `$Name -replace '/', '\'
  `$parts = `$norm -split '\\'
  if (`$parts.Count -ge 2 -and `$parts[0] -eq '45') { return `$parts[1] }
  return ''
}

function Get-EvalName {
  param([string]`$Name)
  if (`$Name -match 'full_eval\\(?<eval>[^\\]+)\\summary\.json$') { return `$Matches.eval }
  return ''
}

function Flatten-Json {
  param(
    [object]`$Node,
    [string]`$Prefix
  )
  `$rows = New-Object System.Collections.Generic.List[object]
  if (`$null -eq `$Node) {
    `$rows.Add([pscustomobject]@{ path = `$Prefix; value = '' })
    return @(`$rows.ToArray())
  }
  if (`$Node -is [System.Management.Automation.PSCustomObject]) {
    foreach (`$prop in `$Node.PSObject.Properties) {
      `$next = if ([string]::IsNullOrWhiteSpace(`$Prefix)) { `$prop.Name } else { "`$Prefix.`$(`$prop.Name)" }
      foreach (`$child in (Flatten-Json -Node `$prop.Value -Prefix `$next)) { `$rows.Add(`$child) }
    }
    return @(`$rows.ToArray())
  }
  if (`$Node -is [System.Collections.IEnumerable] -and -not (`$Node -is [string])) {
    `$i = 0
    foreach (`$item in `$Node) {
      foreach (`$child in (Flatten-Json -Node `$item -Prefix "`$Prefix[`$i]")) { `$rows.Add(`$child) }
      `$i++
      if (`$i -ge 20) { break }
    }
    if (`$i -eq 0) { `$rows.Add([pscustomobject]@{ path = `$Prefix; value = '' }) }
    return @(`$rows.ToArray())
  }
  `$rows.Add([pscustomobject]@{ path = `$Prefix; value = [string]`$Node })
  return @(`$rows.ToArray())
}

`$entries = @(Parse-UnrarListing -ArchivePath `$archivePath | Where-Object { -not `$_.is_dir -and `$_.name -match '\\summary\.json$' } | Sort-Object name)
`$out = New-Object System.Collections.Generic.List[object]
foreach (`$entry in `$entries) {
  `$raw = & `$unrar p -inul `$archivePath `$entry.name 2>&1
  `$exit = `$LASTEXITCODE
  `$text = (`$raw -join "`n")
  try {
    `$json = `$text | ConvertFrom-Json
    `$flat = @(Flatten-Json -Node `$json -Prefix '')
    `$selected = @(`$flat | Where-Object { `$_.path -match '(clip|lpips|fid|kid|time|wall|seconds|sample|count|style|content|overall|mean|avg|accuracy|correct)' })
    foreach (`$m in `$selected) {
      `$out.Add([pscustomobject]@{
        remote_root = `$root
        archive = `$archiveRel
        run = Get-RunName -Name `$entry.name
        eval_name = Get-EvalName -Name `$entry.name
        summary_entry = `$entry.name
        summary_bytes = [string][int64]`$entry.size
        unrar_p_exit_code = [string]`$exit
        metric_path = `$m.path
        metric_value = `$m.value
      })
    }
  } catch {
    `$out.Add([pscustomobject]@{
      remote_root = `$root
      archive = `$archiveRel
      run = Get-RunName -Name `$entry.name
      eval_name = Get-EvalName -Name `$entry.name
      summary_entry = `$entry.name
      summary_bytes = [string][int64]`$entry.size
      unrar_p_exit_code = [string]`$exit
      metric_path = 'parse_error'
      metric_value = `$_.Exception.Message
    })
  }
}
`$out | ConvertTo-Csv -NoTypeInformation
"@

& scp -P $Port -o LogLevel=ERROR $LocalUnrar "${Remote}:C:/Users/Administrator/AppData/Local/Temp/codex_UnRAR.exe" | Out-Null
$LocalRemoteScript = Join-Path $env:TEMP $RemoteScriptName
$RemoteScript | Set-Content -Path $LocalRemoteScript -Encoding UTF8
& scp -P $Port -o LogLevel=ERROR $LocalRemoteScript "${Remote}:$RemoteTempPath" | Out-Null

$raw = & ssh -p $Port -o LogLevel=ERROR $Remote "powershell -NoProfile -ExecutionPolicy Bypass -File `"$RemoteTempWinPath`""
$csvLines = @($raw | Where-Object { $_ -match '^"' })
if (-not $csvLines -or $csvLines.Count -lt 2) {
    $joined = ($raw -join "`n")
    throw "Remote 45.rar summary metric extraction did not return CSV. Output: $joined"
}
$csvLines | Set-Content -Path $LocalOut -Encoding UTF8

& ssh -p $Port -o LogLevel=ERROR $Remote "powershell -NoProfile -Command `"Remove-Item -LiteralPath '$RemoteTempWinPath' -Force -ErrorAction SilentlyContinue; Remove-Item -LiteralPath '$RemoteUnrarPath' -Force -ErrorAction SilentlyContinue`"" | Out-Null
Write-Host "Wrote $LocalOut"
