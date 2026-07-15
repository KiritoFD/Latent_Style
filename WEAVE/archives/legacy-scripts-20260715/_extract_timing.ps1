$ErrorActionPreference = "Continue"

function Show-TrainTiming($dir) {
  Write-Host "==== TRAIN $dir ===="
  $logs = @()
  $logs += Get-ChildItem -LiteralPath $dir -Filter "*.log" -ErrorAction SilentlyContinue
  $logs += Get-ChildItem -LiteralPath (Join-Path $dir "logs") -Filter "*.log" -ErrorAction SilentlyContinue
  $logs += Get-ChildItem -LiteralPath (Join-Path $dir "logs") -Filter "*.csv" -ErrorAction SilentlyContinue
  $logs += Get-ChildItem -LiteralPath (Join-Path $dir "logs") -Filter "training_*.csv" -ErrorAction SilentlyContinue
  foreach ($f in $logs) {
    Write-Host "-- file $($f.FullName) len=$($f.Length)"
  }
  $csv = Get-ChildItem -LiteralPath (Join-Path $dir "logs") -Filter "training_*.csv" -ErrorAction SilentlyContinue | Select-Object -First 1
  if ($csv) {
    $rows = Import-Csv $csv.FullName
    Write-Host "csv rows=$($rows.Count) cols=$($rows[0].PSObject.Properties.Name -join ',')"
    $epochCols = $rows[0].PSObject.Properties.Name | Where-Object { $_ -match 'epoch|time|sec|wall|duration' }
    Write-Host "time-like cols: $($epochCols -join ', ')"
    # print first/last few
    $rows | Select-Object -First 3 | Format-List
    $rows | Select-Object -Last 3 | Format-List
    if ($rows[0].PSObject.Properties.Name -contains 'epoch_sec') {
      $sum = ($rows | Measure-Object -Property epoch_sec -Sum).Sum
      Write-Host "sum epoch_sec=$sum"
    }
    if ($rows[0].PSObject.Properties.Name -contains 'epoch') {
      # try common names
    }
  }
  # grep epoch lines
  $err = Get-ChildItem -LiteralPath (Join-Path $dir "logs") -Filter "*err*" -ErrorAction SilentlyContinue
  $all = @()
  $all += Get-ChildItem -LiteralPath (Join-Path $dir "logs") -File -ErrorAction SilentlyContinue
  foreach ($f in $all) {
    $hits = Select-String -Path $f.FullName -Pattern "Epoch \d+/\d+ \|.*epoch=|Training completed|Saved checkpoint|epoch=" -ErrorAction SilentlyContinue | Select-Object -Last 15
    if ($hits) {
      Write-Host "hits in $($f.Name):"
      $hits | ForEach-Object { $_.Line }
    }
  }
}

function Show-EvalTiming($path) {
  Write-Host "==== EVAL $path ===="
  if (-not (Test-Path $path)) { Write-Host "missing"; return }
  $j = Get-Content $path -Raw | ConvertFrom-Json
  if ($j.timings_sec) {
    Write-Host "timings_sec:"
    $j.timings_sec | ConvertTo-Json -Compress
  }
  if ($j.settings) { Write-Host "settings keys ok" }
  # common fields
  foreach ($k in @('wall_sec','wall_total_sec','generation_sec','timing_lancet_generation_sec','total_sec','infer_sec')) {
    if ($j.PSObject.Properties.Name -contains $k) { Write-Host "$k=$($j.$k)" }
  }
  if ($j.timings_sec) {
    $j.timings_sec.PSObject.Properties | ForEach-Object { Write-Host ("  {0}={1}" -f $_.Name, $_.Value) }
  }
}

# main paper-ish runs
$cands = @(
  "I:\Github\Latent_Style\SchrodingerBridge\exp\dino_s_break\brk_a_ll03_10ep",
  "I:\Github\Latent_Style\SchrodingerBridge\exp\model_probe\target_hf_subband_ft6",
  "I:\Github\Latent_Style\SchrodingerBridge\exp\model_probe\target_hf_content_anchor_ft6",
  "I:\Github\Latent_Style\SchrodingerBridge\exp\model_probe\target_hf_delta_strong_ft6",
  "I:\Github\Latent_Style\SchrodingerBridge\exp\refactor_clean_baseline"
)
foreach ($d in $cands) {
  if (Test-Path $d) { Show-TrainTiming $d } else { Write-Host "MISS $d" }
}

$evals = @(
  "I:\Github\Latent_Style\SchrodingerBridge\exp\dino_s_break\brk_a_ll03_10ep_d5_eval\full_eval\epoch_0010\summary.json",
  "I:\Github\Latent_Style\SchrodingerBridge\exp\model_probe\target_hf_subband_ft6\full_eval\epoch_0006_adain15\summary.json",
  "I:\Github\Latent_Style\SchrodingerBridge\exp\model_probe\target_hf_content_anchor_ft6\full_eval\adain15\summary.json"
)
foreach ($e in $evals) { Show-EvalTiming $e }

# also local G path if mirrored
Write-Host "==== local brk_a train logs if any ===="
