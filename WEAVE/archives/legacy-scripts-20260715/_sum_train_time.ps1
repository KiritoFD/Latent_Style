$ErrorActionPreference = "Continue"
$runs = @(
  "I:\Github\Latent_Style\SchrodingerBridge\exp\dino_s_break\brk_a_ll03_10ep",
  "I:\Github\Latent_Style\SchrodingerBridge\exp\model_probe\target_hf_subband_ft6",
  "I:\Github\Latent_Style\SchrodingerBridge\exp\model_probe\target_hf_content_anchor_ft6",
  "I:\Github\Latent_Style\SchrodingerBridge\exp\model_probe\target_hf_delta_strong_ft6",
  "I:\Github\Latent_Style\SchrodingerBridge\exp\refactor_clean_baseline"
)
foreach ($d in $runs) {
  Write-Host "==== $d ===="
  if (-not (Test-Path $d)) { Write-Host "MISS"; continue }
  $csv = Get-ChildItem -LiteralPath (Join-Path $d "logs") -Filter "training_*.csv" -ErrorAction SilentlyContinue | Sort-Object LastWriteTime -Descending | Select-Object -First 1
  if (-not $csv) { Write-Host "no training csv"; continue }
  Write-Host "csv=$($csv.Name)"
  $rows = Import-Csv $csv.FullName
  $n = $rows.Count
  $sum = 0.0
  $eps = @()
  foreach ($r in $rows) {
    $t = [double]$r.epoch_time_sec
    $sum += $t
    $eps += ("ep{0}:{1:N2}s bs={2} sps={3:N1}" -f $r.epoch, $t, $r.effective_batch_size, $r.samples_per_sec)
  }
  Write-Host ("epochs={0} sum_epoch_time={1:N2}s ({2:N2} min)" -f $n, $sum, ($sum/60.0))
  $eps | ForEach-Object { Write-Host $_ }
  # file mtime span of checkpoints
  $pts = Get-ChildItem -LiteralPath $d -Filter "epoch_*.pt" -ErrorAction SilentlyContinue | Sort-Object LastWriteTime
  if ($pts) {
    $span = ($pts[-1].LastWriteTime - $pts[0].LastWriteTime).TotalSeconds
    Write-Host ("ckpt_span first={0} last={1} span={2:N1}s" -f $pts[0].Name, $pts[-1].Name, $span)
  }
}
# eval summary wall
Write-Host "==== EVAL WALLS ===="
$evals = Get-ChildItem -LiteralPath "I:\Github\Latent_Style\SchrodingerBridge\exp" -Recurse -Filter "summary.json" -ErrorAction SilentlyContinue |
  Where-Object { $_.FullName -match "brk_a_ll03|target_hf_subband_ft6|target_hf_content_anchor|target_hf_delta_strong" }
foreach ($e in $evals) {
  try {
    $j = Get-Content $e.FullName -Raw | ConvertFrom-Json
    if ($j.timings_sec) {
      $w = $j.timings_sec.wall_total
      $g = $j.timings_sec.lancet_generation
      $v = $j.timings_sec.vae_decode
      $u = $j.timings_sec.uint8_cpu_copy
      Write-Host ("{0}`n  wall={1:N1}s gen={2:N1}s vae={3:N1}s uint8={4:N1}s" -f $e.FullName, $w, $g, $v, $u)
    }
  } catch {}
}
