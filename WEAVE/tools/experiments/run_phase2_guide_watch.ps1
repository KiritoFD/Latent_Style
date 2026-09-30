$ErrorActionPreference = "Stop"

$ScriptRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$ToolsRoot = Split-Path -Parent $ScriptRoot
$SbRoot = Split-Path -Parent $ToolsRoot
$WorkspaceRoot = Split-Path -Parent $SbRoot
Set-Location $WorkspaceRoot

$Guide = Join-Path $SbRoot "docs\612-phase2\guide_for_running_codex.md"
$Phase2Snapshot = Join-Path $SbRoot "docs\experiments\phase2_queue_state_snapshot.json"
$OutputRoot = Join-Path $SbRoot "_codex_tmp\phase2_guide_watch"
$StatusMd = Join-Path $OutputRoot "guide_watch_status.md"
$StateJson = Join-Path $OutputRoot "guide_watch_state.json"
$HistoryJsonl = Join-Path $OutputRoot "guide_watch_history.jsonl"
$LogPath = Join-Path $SbRoot "aaai2027\phase2_guide_watcher.log"

New-Item -ItemType Directory -Force -Path $OutputRoot | Out-Null
New-Item -ItemType Directory -Force -Path (Split-Path -Parent $LogPath) | Out-Null

$Python = (Get-Command python -CommandType Application -ErrorAction Stop | Select-Object -First 1).Source
$Script = Join-Path $ScriptRoot "refresh_phase2_guide_watch.py"

Add-Content -LiteralPath $LogPath -Value ("=== GUIDE WATCH START " + (Get-Date -Format o) + " ===")
& $Python $Script `
  --guide $Guide `
  --phase2-snapshot $Phase2Snapshot `
  --status-md $StatusMd `
  --state-json $StateJson `
  --history-jsonl $HistoryJsonl *>> $LogPath
$rc = $LASTEXITCODE
Add-Content -LiteralPath $LogPath -Value ("=== GUIDE WATCH END rc=" + $rc + " " + (Get-Date -Format o) + " ===")
exit $rc
