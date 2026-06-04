param(
  [string]$CandidateCsv = "EXPERIMENT_ARCHAEOLOGY\_scratch_raw_scans\cleanup_checkpoint_cleanup_candidates.csv",
  [string]$OutDir = "EXPERIMENT_ARCHAEOLOGY\cleanup",
  [string]$Root = "."
)

$ErrorActionPreference = "SilentlyContinue"
$resolvedRoot = (Resolve-Path -LiteralPath $Root).Path
$resolvedOut = Join-Path $resolvedRoot $OutDir
New-Item -ItemType Directory -Force -Path $resolvedOut | Out-Null

$allowedExt = @(".pt", ".pth", ".ckpt", ".model", ".pkl", ".npz")
$rows = New-Object System.Collections.Generic.List[object]
$summary = [ordered]@{
  candidate_csv = $CandidateCsv
  root = $resolvedRoot
  deleted_count = 0
  deleted_mb = 0.0
  skipped_count = 0
  failed_count = 0
}

Import-Csv -LiteralPath (Join-Path $resolvedRoot $CandidateCsv) | ForEach-Object {
  $relPath = [string]$_.checkpoint_path
  $class = [string]$_.cleanup_class
  $sizeMb = [double]($_.size_mb)
  $fullPath = Join-Path $resolvedRoot $relPath
  $action = "skipped"
  $reason = ""

  if ($class -ne "likely_non_mainline_delete_candidate") {
    $reason = "cleanup_class_not_explicit_non_mainline"
  } elseif (-not ($allowedExt -contains ([System.IO.Path]::GetExtension($fullPath).ToLower()))) {
    $reason = "extension_guard"
  } elseif (-not ((Resolve-Path -LiteralPath (Split-Path -Parent $fullPath) -ErrorAction SilentlyContinue).Path.StartsWith($resolvedRoot, [System.StringComparison]::OrdinalIgnoreCase))) {
    $reason = "outside_root_guard"
  } elseif (-not (Test-Path -LiteralPath $fullPath -PathType Leaf)) {
    $reason = "missing_before_delete"
  } else {
    Remove-Item -LiteralPath $fullPath -Force -ErrorAction SilentlyContinue
    if (Test-Path -LiteralPath $fullPath -PathType Leaf) {
      $action = "failed"
      $reason = "remove_item_failed"
      $summary.failed_count += 1
    } else {
      $action = "deleted"
      $reason = "deleted_likely_non_mainline_checkpoint"
      $summary.deleted_count += 1
      $summary.deleted_mb = [Math]::Round(([double]$summary.deleted_mb + $sizeMb), 3)
    }
  }

  if ($action -eq "skipped") {
    $summary.skipped_count += 1
  }

  $rows.Add([pscustomobject]@{
    checkpoint_path = $relPath
    action = $action
    size_mb = $_.size_mb
    cleanup_class = $class
    reason = $reason
  })
}

$rows | Export-Csv -NoTypeInformation -Encoding UTF8 -Path (Join-Path $resolvedOut "local_deleted_checkpoints.csv")
[pscustomobject]$summary | ConvertTo-Json | Set-Content -Encoding UTF8 -Path (Join-Path $resolvedOut "local_delete_summary.json")
Get-Content -LiteralPath (Join-Path $resolvedOut "local_delete_summary.json")
