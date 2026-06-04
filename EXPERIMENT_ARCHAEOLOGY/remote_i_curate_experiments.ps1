param(
  [string]$Root = "I:\",
  [string]$OutDir = "I:\latent_style_remote_curated"
)

$ErrorActionPreference = "SilentlyContinue"
New-Item -ItemType Directory -Force -Path $OutDir | Out-Null

function Guess-Dataset([string]$s) {
  $t = $s.ToLower()
  if ($t.Contains("distinct5") -or $t.Contains("wikiart_distinct5")) { return "distinct5_512" }
  if ($t.Contains("wikiart512") -or $t.Contains("wikiart_512") -or $t.Contains("wikiart_latents_512")) { return "wikiart512_5style" }
  if ($t.Contains("legacy256") -or $t.Contains("overfit50") -or $t.Contains("protocol_a_800") -or $t.Contains("latent-256")) { return "legacy256_overfit50" }
  if ($t.Contains("complete_750") -or $t.Contains("protocol750") -or $t.Contains("strict_750") -or $t.Contains("formal_eval_750")) { return "strict_protocol_750" }
  if ($t.Contains("run_511")) { return "run511_5domain" }
  if ($t.Contains("5x5") -or $t.Contains("cut_5x5") -or $t.Contains("sdedit_multi") -or $t.Contains("photo_to_monet")) { return "photo_monet_5x5" }
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

function Guess-Resolution([string]$s) {
  $t = $s.ToLower()
  if ($t.Contains("512")) { return "512" }
  if ($t.Contains("256")) { return "256" }
  return ""
}

function Add-Row($rows, $period, $method, $dataset, $resolution, $run, $scope, $images, $clipStyle, $lpips, $clipContent, $trainValue, $trainUnit, $trainLabel, $inferValue, $inferUnit, $inferLabel, $sourcePath, $sourceKind, $status, $note) {
  $rows.Add([pscustomobject]@{
    period = $period
    source_root = $Root
    method = $method
    dataset_or_setting = $dataset
    dataset_key = $dataset
    resolution = $resolution
    variant_or_run = $run
    scope = $scope
    images = $images
    clip_style = $clipStyle
    content_lpips = $lpips
    clip_content = $clipContent
    train_time_value = $trainValue
    train_time_unit = $trainUnit
    train_time_label = $trainLabel
    infer_time_value = $inferValue
    infer_time_unit = $inferUnit
    infer_time_label = $inferLabel
    params_m = ""
    hardware = ""
    status = $status
    source_path = $sourcePath
    source_kind = $sourceKind
    note = $note
  })
}

$rows = New-Object System.Collections.Generic.List[object]
$timeline = New-Object System.Collections.Generic.List[object]
$dirs = @{}

Get-ChildItem -LiteralPath $Root -Filter "summary.json" -Recurse -File -Force | ForEach-Object {
  $p = $_.FullName
  $pathLower = $p.ToLower()
  if ($pathLower.Contains("\.git\") -or $pathLower.Contains("\node_modules\") -or $pathLower.Contains("\__pycache__\")) { return }
  $isExperiment = $pathLower.Contains("\full_eval\") -or $pathLower.Contains("\formal_eval") -or $pathLower.Contains("\eval_") -or $pathLower.Contains("\outputs\") -or $pathLower.Contains("\runs\") -or $pathLower.Contains("\exp\") -or $pathLower.Contains("\docs\experiments\")
  if (-not $isExperiment) { return }
  $json = Get-Content -LiteralPath $p -Raw | ConvertFrom-Json
  $overall = $json.overall
  if ($null -eq $overall) { $overall = $json }
  $timings = $json.timings_sec
  $clipStyle = ""
  $lpips = ""
  $clipContent = ""
  $images = ""
  $inferValue = ""
  $inferUnit = ""
  $inferLabel = ""
  if ($overall.clip_style -ne $null) { $clipStyle = [string]$overall.clip_style }
  if ($overall.clip_style_all -ne $null) { $clipStyle = [string]$overall.clip_style_all }
  if ($overall.content_lpips -ne $null) { $lpips = [string]$overall.content_lpips }
  if ($overall.content_lpips_all -ne $null) { $lpips = [string]$overall.content_lpips_all }
  if ($overall.lpips -ne $null) { $lpips = [string]$overall.lpips }
  if ($overall.clip_content -ne $null) { $clipContent = [string]$overall.clip_content }
  if ($json.count -ne $null) { $images = [string]$json.count }
  if ($json.images -ne $null) { $images = [string]$json.images }
  if ($timings.wall_total -ne $null) {
    $inferValue = [string]$timings.wall_total
    $inferUnit = "s"
    $inferLabel = "$inferValue s timings_sec.wall_total"
  }
  $dataset = Guess-Dataset $p
  $method = Guess-Method $p
  $runDir = Split-Path -Parent $p
  Add-Row $rows $_.LastWriteTime.ToString("yyyy-MM-dd") $method $dataset (Guess-Resolution $p) (Split-Path -Leaf $runDir) "full_eval_or_summary" $images $clipStyle $lpips $clipContent "" "" "" $inferValue $inferUnit $inferLabel $p "remote_summary_json_curated" "summary_found" "Remote curated summary.json after experiment-path filter."
  $dirs[$runDir] = $true
}

foreach ($pattern in @("training_*.csv", "remote_train.log", "train.log", "loss_log.txt", "launcher_stdout.log", "watch_eval.log", "generate.log", "eval.log")) {
  Get-ChildItem -LiteralPath $Root -Filter $pattern -Recurse -File -Force | ForEach-Object {
    $p = $_.FullName
    $pathLower = $p.ToLower()
    if ($pathLower.Contains("\.git\") -or $pathLower.Contains("\node_modules\") -or $pathLower.Contains("\__pycache__\")) { return }
    $isExperiment = $pathLower.Contains("\logs\") -or $pathLower.Contains("\exp\") -or $pathLower.Contains("\runs\") -or $pathLower.Contains("\outputs\") -or $pathLower.Contains("\baseline_pipeline\") -or $pathLower.Contains("\schrodingerbridge\")
    if (-not $isExperiment) { return }
    $text = ""
    if ($_.Length -lt 8388608) {
      $text = Get-Content -LiteralPath $p -Raw
    }
    $elapsed = ""
    $elapsedLabel = ""
    $m = [regex]::Match($text, "elapsed_sec\s*=\s*(\d+(?:\.\d+)?)|elapsed_sec\s+(\d+(?:\.\d+)?)|wall_total['""]?\s*[:=]\s*(\d+(?:\.\d+)?)", "IgnoreCase")
    if ($m.Success) {
      foreach ($g in $m.Groups) {
        if ($g.Value -match "^\d+(\.\d+)?$") { $elapsed = $g.Value; break }
      }
      if ($elapsed) { $elapsedLabel = "$elapsed s regex log hit" }
    }
    $dataset = Guess-Dataset $p
    $method = Guess-Method $p
    $runDir = Split-Path -Parent $p
    Add-Row $rows $_.LastWriteTime.ToString("yyyy-MM-dd") $method $dataset (Guess-Resolution $p) (Split-Path -Leaf $runDir) "train_or_eval_log" "" "" "" "" $elapsed "s" $elapsedLabel "" "" "" $p "remote_training_log_curated" "log_found" "Remote curated train/eval log after experiment-path filter."
    $timeline.Add([pscustomobject]@{
      period = $_.LastWriteTime.ToString("yyyy-MM-ddTHH:mm:ss")
      source_root = $Root
      dataset_guess = $dataset
      method_guess = $method
      event_type = "log_file"
      path = $p
      elapsed_sec_hint = $elapsed
      note = "Curated remote experiment log."
    })
    $dirs[$runDir] = $true
  }
}

$dirRows = New-Object System.Collections.Generic.List[object]
foreach ($d in $dirs.Keys) {
  $dirRows.Add([pscustomobject]@{
    directory = $d
    source_root = $Root
    dataset_guess = Guess-Dataset $d
    method_guess = Guess-Method $d
    status = "curated_experiment_dir"
    note = "Directory retained because curated summary/log evidence was found."
  })
}

$ckptRows = New-Object System.Collections.Generic.List[object]
foreach ($pattern in @("*.pt", "*.pth", "*.ckpt", "*.model", "*.pkl", "*.npz")) {
  Get-ChildItem -LiteralPath $Root -Filter $pattern -Recurse -File -Force | ForEach-Object {
    $p = $_.FullName
    $lower = $p.ToLower()
    if ($lower.Contains("\.git\") -or $lower.Contains("\node_modules\") -or $lower.Contains("\__pycache__\")) { return }
    $class = "review_delete_candidate"
    if ($lower.Contains("aaai2027") -or $lower.Contains("distinct5_512") -or $lower.Contains("s-add__k-1_c-0_w-20_col-0") -or $lower.Contains("local_wsl_wikiart512_hist_b32_e8")) {
      $class = "likely_mainline_keep"
    } elseif ($lower.Contains("smoke") -or $lower.Contains("tmp") -or $lower.Contains("archive") -or $lower.Contains("old_experiment_dirs") -or $lower.Contains("\run_511\outputs\")) {
      $class = "non_mainline_delete_candidate"
    }
    $ckptRows.Add([pscustomobject]@{
      checkpoint_path = $p
      source_root = $Root
      size_mb = [Math]::Round($_.Length / 1MB, 3)
      modified = $_.LastWriteTime.ToString("s")
      dataset_guess = Guess-Dataset $p
      method_guess = Guess-Method $p
      cleanup_class = $class
      note = "Curated cleanup manifest only; not deleted by this script."
    })
  }
}

$rows | Export-Csv -NoTypeInformation -Encoding UTF8 -Path (Join-Path $OutDir "remote_i_curated_experiments.csv")
$timeline | Export-Csv -NoTypeInformation -Encoding UTF8 -Path (Join-Path $OutDir "remote_i_timeline.csv")
$dirRows | Export-Csv -NoTypeInformation -Encoding UTF8 -Path (Join-Path $OutDir "remote_i_curated_directory_index.csv")
$ckptRows | Export-Csv -NoTypeInformation -Encoding UTF8 -Path (Join-Path $OutDir "remote_i_checkpoint_cleanup_candidates.csv")
[pscustomobject]@{
  root = $Root
  out_dir = $OutDir
  curated_rows = $rows.Count
  timeline_rows = $timeline.Count
  curated_dirs = $dirRows.Count
  checkpoint_candidates = $ckptRows.Count
} | ConvertTo-Json | Set-Content -Encoding UTF8 -Path (Join-Path $OutDir "remote_i_curated_summary.json")
Get-Content -LiteralPath (Join-Path $OutDir "remote_i_curated_summary.json")
