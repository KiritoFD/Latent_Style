param(
  [string]$CuratedDir = "I:\latent_style_remote_curated"
)

$ErrorActionPreference = "SilentlyContinue"

function Ensure-Dir([string]$p) {
  New-Item -ItemType Directory -Force -Path $p | Out-Null
}

function Write-Lines([string]$Path, [string[]]$Lines) {
  $Lines | Set-Content -Encoding UTF8 -Path $Path
}

Ensure-Dir $CuratedDir
$byDataset = Join-Path $CuratedDir "by_dataset"
Ensure-Dir $byDataset

$experimentsPath = Join-Path $CuratedDir "remote_i_curated_experiments.csv"
$timelinePath = Join-Path $CuratedDir "remote_i_timeline.csv"
$deletePath = Join-Path $CuratedDir "remote_i_deleted_checkpoints.csv"

$experiments = @()
$timeline = @()
$deleted = @()
if (Test-Path -LiteralPath $experimentsPath) { $experiments = Import-Csv -LiteralPath $experimentsPath }
if (Test-Path -LiteralPath $timelinePath) { $timeline = Import-Csv -LiteralPath $timelinePath }
if (Test-Path -LiteralPath $deletePath) { $deleted = Import-Csv -LiteralPath $deletePath }

$experiments | Group-Object dataset_key | ForEach-Object {
  $dataset = if ($_.Name) { $_.Name } else { "unknown" }
  $safe = ($dataset -replace "[^A-Za-z0-9_.-]+", "_").Trim("_")
  if (-not $safe) { $safe = "unknown" }
  $_.Group | Export-Csv -NoTypeInformation -Encoding UTF8 -Path (Join-Path $byDataset "$safe.csv")
}

$deletedActual = @($deleted | Where-Object { $_.action -eq "deleted" })
$deletedMb = 0.0
foreach ($d in $deletedActual) {
  $v = 0.0
  [void][double]::TryParse([string]$d.size_mb, [ref]$v)
  $deletedMb += $v
}
$deletedMb = [Math]::Round($deletedMb, 3)

$datasetGroups = $experiments | Group-Object dataset_key | Sort-Object Count -Descending
$methodGroups = $experiments | Group-Object method | Sort-Object Count -Descending
$sourceGroups = $experiments | Group-Object source_kind | Sort-Object Count -Descending

$periodGroups = $experiments |
  Where-Object { $_.period } |
  Group-Object period |
  Sort-Object Name

$timelineGroups = $timeline |
  Where-Object { $_.period } |
  Sort-Object period |
  Select-Object -First 400

$lines = New-Object System.Collections.Generic.List[string]
$lines.Add("# Remote I Drive Experiment Archaeology")
$lines.Add("")
$lines.Add("Generated on remote host from I:\ after curated experiment filtering and non-mainline checkpoint cleanup.")
$lines.Add("")
$lines.Add("## Scope")
$lines.Add("")
$lines.Add("- Source root: I:\")
$lines.Add("- Curated experiment rows: $($experiments.Count)")
$lines.Add("- Timeline rows: $($timeline.Count)")
$lines.Add("- Deleted non-mainline checkpoints: $($deletedActual.Count)")
$lines.Add("- Deleted checkpoint size: $deletedMb MB")
$lines.Add("")
$lines.Add("## Dataset Row Counts")
$lines.Add("")
foreach ($g in $datasetGroups) { $lines.Add("- " + $g.Name + ": " + $g.Count) }
$lines.Add("")
$lines.Add("## Method Row Counts")
$lines.Add("")
foreach ($g in ($methodGroups | Select-Object -First 40)) {
  $name = if ($g.Name) { $g.Name } else { "unknown" }
  $lines.Add("- " + $name + ": " + $g.Count)
}
$lines.Add("")
$lines.Add("## Source Kinds")
$lines.Add("")
foreach ($g in $sourceGroups) { $lines.Add("- " + $g.Name + ": " + $g.Count) }
$lines.Add("")
$lines.Add("## Period / Timeline Skeleton")
$lines.Add("")
foreach ($g in $periodGroups) {
  $datasets = ($g.Group | Group-Object dataset_key | Sort-Object Count -Descending | Select-Object -First 5 | ForEach-Object { "$($_.Name):$($_.Count)" }) -join ", "
  $methods = ($g.Group | Group-Object method | Sort-Object Count -Descending | Select-Object -First 5 | ForEach-Object { "$(if ($_.Name) {$_.Name} else {'unknown'}):$($_.Count)" }) -join ", "
  $lines.Add("- " + $g.Name + ": " + $g.Count + " rows; datasets {" + $datasets + "}; methods {" + $methods + "}")
}
$lines.Add("")
$lines.Add("## First Timeline Events")
$lines.Add("")
foreach ($r in $timelineGroups) {
  $lines.Add("- " + $r.period + ": " + $r.event_type + " dataset=" + $r.dataset_guess + " method=" + $r.method_guess + " path=" + $r.path + " elapsed_sec_hint=" + $r.elapsed_sec_hint)
}
$lines.Add("")
$lines.Add("## Cleanup Notes")
$lines.Add("")
$lines.Add("Only rows marked non_mainline_delete_candidate were deleted. likely_mainline_keep and review_delete_candidate were retained.")
$lines.Add("The full per-file deletion audit is remote_i_deleted_checkpoints.csv.")
Write-Lines (Join-Path $CuratedDir "REMOTE_I_EXPERIMENT_LOG.md") $lines.ToArray()

$indexLines = New-Object System.Collections.Generic.List[string]
$indexLines.Add("# Remote I Curated Index")
$indexLines.Add("")
$indexLines.Add("- remote_i_curated_experiments.csv: filtered experiment rows from summary/log evidence.")
$indexLines.Add("- remote_i_timeline.csv: curated timeline from train/eval logs.")
$indexLines.Add("- remote_i_curated_directory_index.csv: directories retained by curated evidence.")
$indexLines.Add("- by_dataset/*.csv: one file per dataset key.")
$indexLines.Add("- remote_i_deleted_checkpoints.csv: per-checkpoint deletion/skipping audit.")
$indexLines.Add("- remote_i_delete_summary.json: cleanup aggregate.")
$indexLines.Add("- REMOTE_I_EXPERIMENT_LOG.md: narrative/timeline summary.")
$indexLines.Add("")
$indexLines.Add("## Deletion Summary")
$indexLines.Add("")
$indexLines.Add("- Deleted checkpoints: $($deletedActual.Count)")
$indexLines.Add("- Deleted MB: $deletedMb")
Write-Lines (Join-Path $CuratedDir "README_REMOTE_I.md") $indexLines.ToArray()

[pscustomobject]@{
  curated_dir = $CuratedDir
  by_dataset_files = (Get-ChildItem -LiteralPath $byDataset -Filter "*.csv" | Measure-Object).Count
  experiment_rows = $experiments.Count
  timeline_rows = $timeline.Count
  deleted_checkpoints = $deletedActual.Count
  deleted_mb = $deletedMb
} | ConvertTo-Json | Set-Content -Encoding UTF8 -Path (Join-Path $CuratedDir "remote_i_narrative_summary.json")

Get-Content -LiteralPath (Join-Path $CuratedDir "remote_i_narrative_summary.json")
