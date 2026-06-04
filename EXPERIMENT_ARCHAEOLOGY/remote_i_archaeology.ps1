param(
  [string]$Root = "I:\",
  [string]$OutDir = "$env:TEMP\latent_style_remote_archaeology"
)

$ErrorActionPreference = "SilentlyContinue"
New-Item -ItemType Directory -Force -Path $OutDir | Out-Null

$evidenceExt = @(".csv", ".json", ".jsonl", ".md", ".txt", ".log", ".out", ".err", ".yaml", ".yml")
$ckptExt = @(".pt", ".pth", ".ckpt", ".model", ".pkl", ".npz")
$skipNames = @(".git", "__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache", "node_modules")

function Guess-Dataset([string]$s) {
  $t = $s.ToLower()
  if ($t.Contains("distinct5") -or $t.Contains("wikiart_distinct5")) { return "distinct5_512" }
  if ($t.Contains("wikiart512") -or $t.Contains("wikiart_512")) { return "wikiart512_5style" }
  if ($t.Contains("legacy256") -or $t.Contains("overfit50") -or $t.Contains("protocol_a_800") -or $t.Contains("latent-256")) { return "legacy256_overfit50" }
  if ($t.Contains("complete_750") -or $t.Contains("protocol750") -or $t.Contains("strict_750")) { return "strict_protocol_750" }
  if ($t.Contains("run_511")) { return "run511_5domain" }
  if ($t.Contains("5x5") -or $t.Contains("cut_5x5") -or $t.Contains("sdedit_multi")) { return "photo_monet_5x5" }
  if ($t.Contains("seedream")) { return "seedream_wikiart512" }
  return "unknown"
}

function Guess-Method([string]$s) {
  $t = $s.ToLower()
  if ($t.Contains("lancet") -or $t.Contains("lbm") -or $t.Contains("schrodingerbridge") -or $t.Contains("s-add__")) { return "LANCET/LBM" }
  if ($t.Contains("samst")) { return "SaMST" }
  if ($t.Contains("samam")) { return "SaMAM" }
  if ($t.Contains("s2wat")) { return "S2WAT" }
  if ($t.Contains("styleid")) { return "StyleID" }
  if ($t.Contains("adain")) { return "AdaIN" }
  if ($t.Contains("stytr2")) { return "StyTr2" }
  if ($t.Contains("cast")) { return "CAST" }
  if ($t.Contains("aesfa")) { return "AesFA" }
  if ($t.Contains("aespa")) { return "AesPA-Net" }
  if ($t.Contains("cut_")) { return "CUT" }
  if ($t.Contains("cyclegan")) { return "CycleGAN" }
  if ($t.Contains("sdedit")) { return "SDEdit" }
  if ($t.Contains("sdturbo")) { return "SD-Turbo" }
  if ($t.Contains("seedream")) { return "Seedream" }
  if ($t.Contains("idt")) { return "IDT" }
  return ""
}

function Should-Skip([System.IO.FileSystemInfo]$item) {
  foreach ($part in $item.FullName.Split([System.IO.Path]::DirectorySeparatorChar)) {
    if ($skipNames -contains $part) { return $true }
  }
  return $false
}

$evidence = New-Object System.Collections.Generic.List[object]
$checkpoints = New-Object System.Collections.Generic.List[object]
$timingHits = New-Object System.Collections.Generic.List[object]
$dirIndex = @{}

foreach ($pattern in (($evidenceExt + $ckptExt) | ForEach-Object { "*$_" })) {
Get-ChildItem -LiteralPath $Root -Filter $pattern -Recurse -File -Force | ForEach-Object {
  if (Should-Skip $_) { return }
  $ext = $_.Extension.ToLower()
  $dir = $_.DirectoryName
  if (-not $dirIndex.ContainsKey($dir)) {
    $dirIndex[$dir] = [ordered]@{
      directory = $dir
      source_root = $Root
      dataset_guess = Guess-Dataset $dir
      method_guess = Guess-Method $dir
      files_seen = 0
      evidence_files = 0
      checkpoints = 0
    }
  }
  $dirIndex[$dir].files_seen += 1

  if ($evidenceExt -contains $ext) {
    $text = ""
    if ($_.Length -lt 8388608) {
      $text = Get-Content -LiteralPath $_.FullName -Raw -ErrorAction SilentlyContinue
    }
    $timingCount = 0
    $metricCount = 0
    if ($text) {
      $timingCount = ([regex]::Matches($text, "elapsed_sec|train_sec|infer_sec|wall_total|wall_seconds|EVAL_WALL_SECONDS|train_wall", "IgnoreCase")).Count
      $metricCount = ([regex]::Matches($text, "clip_style|content_lpips|lpips|ssim_y|edge_f1|artfid|kid|musiq|maniqa", "IgnoreCase")).Count
      foreach ($m in [regex]::Matches($text, "(elapsed_sec|wall_seconds|EVAL_WALL_SECONDS|wall_total)\s*[:=]\s*(\d+(?:\.\d+)?)", "IgnoreCase")) {
        $timingHits.Add([pscustomobject]@{
          period = $_.LastWriteTime.ToString("s")
          method = Guess-Method $_.FullName
          dataset_or_setting = Guess-Dataset $_.FullName
          timing_key = $m.Groups[1].Value
          timing_value = $m.Groups[2].Value
          timing_unit = "s"
          source_path = $_.FullName
          note = "remote regex timing hit"
        })
      }
    }
    $evidence.Add([pscustomobject]@{
      source_path = $_.FullName
      source_root = $Root
      extension = $ext
      size_bytes = $_.Length
      modified = $_.LastWriteTime.ToString("s")
      dataset_guess = Guess-Dataset ($_.FullName + "`n" + $text.Substring(0, [Math]::Min(1000, $text.Length)))
      method_guess = Guess-Method ($_.FullName + "`n" + $text.Substring(0, [Math]::Min(1000, $text.Length)))
      run_dir = $_.DirectoryName
      timing_hit_count = $timingCount
      metric_hit_count = $metricCount
      note = "remote evidence indexed"
    })
    $dirIndex[$dir].evidence_files += 1
  }

  if ($ckptExt -contains $ext) {
    $class = "review_delete_candidate"
    $lower = $_.FullName.ToLower()
    if ($lower.Contains("s-add__k-1_c-0_w-20_col-0") -or $lower.Contains("local_wsl_wikiart512_hist_b32_e8") -or $lower.Contains("aaai2027") -or $lower.Contains("distinct5_512_20260602")) {
      $class = "likely_mainline_keep"
    } elseif ($lower.Contains("smoke") -or $lower.Contains("tmp") -or $lower.Contains("archive") -or $lower.Contains("old_experiment_dirs") -or $lower.Contains("run_511\outputs")) {
      $class = "likely_non_mainline_delete_candidate"
    }
    $checkpoints.Add([pscustomobject]@{
      checkpoint_path = $_.FullName
      source_root = $Root
      size_mb = [Math]::Round($_.Length / 1MB, 3)
      modified = $_.LastWriteTime.ToString("s")
      dataset_guess = Guess-Dataset $_.FullName
      method_guess = Guess-Method $_.FullName
      cleanup_class = $class
      note = "remote manifest only; not deleted"
    })
    $dirIndex[$dir].checkpoints += 1
  }
}
}

$evidence | Export-Csv -NoTypeInformation -Encoding UTF8 -Path (Join-Path $OutDir "remote_i_evidence_files.csv")
$checkpoints | Export-Csv -NoTypeInformation -Encoding UTF8 -Path (Join-Path $OutDir "remote_i_checkpoint_candidates.csv")
$timingHits | Export-Csv -NoTypeInformation -Encoding UTF8 -Path (Join-Path $OutDir "remote_i_timing_hits.csv")
$dirRows = New-Object System.Collections.Generic.List[object]
foreach ($entry in $dirIndex.GetEnumerator()) {
  $dirRows.Add([pscustomobject]$entry.Value)
}
$dirRows | Export-Csv -NoTypeInformation -Encoding UTF8 -Path (Join-Path $OutDir "remote_i_directory_index.csv")

[pscustomobject]@{
  root = $Root
  out_dir = $OutDir
  evidence_files = $evidence.Count
  checkpoints = $checkpoints.Count
  timing_hits = $timingHits.Count
  directories = $dirIndex.Count
} | ConvertTo-Json | Set-Content -Encoding UTF8 -Path (Join-Path $OutDir "remote_i_summary.json")

Get-Content -LiteralPath (Join-Path $OutDir "remote_i_summary.json")
