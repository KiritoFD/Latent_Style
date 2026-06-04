param(
  [string]$CandidateCsv = "I:\latent_style_remote_curated\remote_i_checkpoint_cleanup_candidates.csv",
  [string]$OutDir = "I:\latent_style_remote_curated",
  [string]$Root = "I:\"
)

$ErrorActionPreference = "SilentlyContinue"
New-Item -ItemType Directory -Force -Path $OutDir | Out-Null

$allowedExt = @(".pt", ".pth", ".ckpt", ".model", ".pkl", ".npz")
$deleteLog = New-Object System.Collections.Generic.List[object]
$summary = [ordered]@{
  candidate_csv = $CandidateCsv
  root = $Root
  deleted_count = 0
  deleted_mb = 0.0
  skipped_count = 0
  failed_count = 0
}

if (-not (Test-Path -LiteralPath $CandidateCsv)) {
  [pscustomobject]@{
    checkpoint_path = ""
    action = "failed"
    size_mb = ""
    cleanup_class = ""
    reason = "candidate csv not found"
  } | Export-Csv -NoTypeInformation -Encoding UTF8 -Path (Join-Path $OutDir "remote_i_deleted_checkpoints.csv")
  [pscustomobject]$summary | ConvertTo-Json | Set-Content -Encoding UTF8 -Path (Join-Path $OutDir "remote_i_delete_summary.json")
  Get-Content -LiteralPath (Join-Path $OutDir "remote_i_delete_summary.json")
  exit 1
}

Import-Csv -LiteralPath $CandidateCsv | ForEach-Object {
  $path = [string]$_.checkpoint_path
  $class = [string]$_.cleanup_class
  $sizeMb = [double]($_.size_mb)
  $reason = ""
  $action = "skipped"

  if ($class -ne "non_mainline_delete_candidate") {
    $reason = "cleanup_class_not_non_mainline"
  } elseif (-not $path.StartsWith($Root, [System.StringComparison]::OrdinalIgnoreCase)) {
    $reason = "outside_root_guard"
  } elseif (-not ($allowedExt -contains ([System.IO.Path]::GetExtension($path).ToLower()))) {
    $reason = "extension_guard"
  } elseif (-not (Test-Path -LiteralPath $path -PathType Leaf)) {
    $reason = "missing_before_delete"
  } else {
    Remove-Item -LiteralPath $path -Force -ErrorAction SilentlyContinue
    if (Test-Path -LiteralPath $path -PathType Leaf) {
      $action = "failed"
      $reason = "remove_item_failed"
      $summary.failed_count += 1
    } else {
      $action = "deleted"
      $reason = "deleted_non_mainline_checkpoint"
      $summary.deleted_count += 1
      $summary.deleted_mb = [Math]::Round(([double]$summary.deleted_mb + $sizeMb), 3)
    }
  }

  if ($action -eq "skipped") {
    $summary.skipped_count += 1
  }

  $deleteLog.Add([pscustomobject]@{
    checkpoint_path = $path
    action = $action
    size_mb = $_.size_mb
    cleanup_class = $class
    reason = $reason
  })
}

$deleteLog | Export-Csv -NoTypeInformation -Encoding UTF8 -Path (Join-Path $OutDir "remote_i_deleted_checkpoints.csv")
[pscustomobject]$summary | ConvertTo-Json | Set-Content -Encoding UTF8 -Path (Join-Path $OutDir "remote_i_delete_summary.json")
Get-Content -LiteralPath (Join-Path $OutDir "remote_i_delete_summary.json")
