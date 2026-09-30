# Pure inference timing: generation only, no metrics.
$ErrorActionPreference = "Continue"
$Root = "I:\Github\Latent_Style\SchrodingerBridge"
Set-Location -LiteralPath $Root
$env:PYTHONPATH = "$Root\src"
$outRoot = "$Root\exp\model_probe\inf_timing"
New-Item -ItemType Directory -Force -Path $outRoot | Out-Null

$runs = @(
  @{
    name = "brk_a_ll03_10ep"
    ckpt = "$Root\exp\dino_s_break\brk_a_ll03_10ep\epoch_0010.pt"
    cfg  = "$Root\exp\dino_s_break\brk_a_ll03_10ep\config.json"
  },
  @{
    name = "target_hf_subband_ft6"
    ckpt = "$Root\exp\model_probe\target_hf_subband_ft6\epoch_0006.pt"
    cfg  = "$Root\exp\model_probe\target_hf_subband_ft6\config.json"
  }
)

$summary = @()
foreach ($r in $runs) {
  if (-not (Test-Path $r.ckpt)) {
    Write-Host "SKIP missing ckpt $($r.ckpt)"
    continue
  }
  $out = Join-Path $outRoot $r.name
  New-Item -ItemType Directory -Force -Path $out | Out-Null
  $log = Join-Path $out "generation_only.log"
  $err = Join-Path $out "generation_only.err"
  Write-Host "==== GEN-ONLY $($r.name) ===="
  $t0 = Get-Date
  # Use run config as base; force AdaIN=1.5 via config_override (same as main-table op point).
  $override = "$Root\configs\eval_adain_15.json"
  & python -u "$Root\src\utils\run_evaluation.py" `
    --config $r.cfg `
    --config_override $override `
    --checkpoint $r.ckpt `
    --output $out `
    --generation_only `
    --profile_timing `
    --batch_size 2 `
    --generation_batch_size 2 `
    --num_steps 8 `
    --save_generated_images `
    --no-save_summary_grid `
    1> $log 2> $err
  $code = $LASTEXITCODE
  $t1 = Get-Date
  $wallPs = ($t1 - $t0).TotalSeconds
  Write-Host "exit=$code wall_ps=$([math]::Round($wallPs,2))s"

  $sj = Get-ChildItem $out -Recurse -Filter "summary.json" -ErrorAction SilentlyContinue | Select-Object -First 1
  $row = [ordered]@{
    name = $r.name
    exit = $code
    wall_ps = [math]::Round($wallPs, 3)
  }
  if ($sj) {
    try {
      $j = Get-Content $sj.FullName -Raw | ConvertFrom-Json
      $row.summary = $sj.FullName
      if ($j.timings_sec) {
        $t = $j.timings_sec
        $row.wall_total = [math]::Round([double]$t.wall_total, 3)
        $row.lancet_generation = [math]::Round([double]$t.lancet_generation, 3)
        $row.vae_decode = [math]::Round([double]$t.vae_decode, 3)
        $row.uint8_cpu_copy = [math]::Round([double]$t.uint8_cpu_copy, 3)
        $row.load_vae = [math]::Round([double]$t.load_vae, 3)
        $row.load_lancet = [math]::Round([double]$t.load_lancet, 3)
        $row.source_latent_cache = [math]::Round([double]$t.source_latent_cache, 3)
        if ($t.wall_total -gt 0) {
          $row.ms_per_img_wall = [math]::Round(1000.0 * [double]$t.wall_total / 750.0, 2)
        }
        if ($t.lancet_generation -gt 0) {
          $row.ms_per_img_gen = [math]::Round(1000.0 * [double]$t.lancet_generation / 750.0, 2)
        }
      }
      if ($j.settings) {
        $row.mode = $j.settings.mode
        $row.num_steps = $j.settings.num_steps
        $row.generation_batch_size = $j.settings.generation_batch_size
      }
    } catch {
      $row.parse_error = "$_"
    }
  } else {
    $row.summary = "missing"
  }
  $summary += [pscustomobject]$row
  $row | ConvertTo-Json -Compress | Write-Host
}

$outJson = Join-Path $outRoot "generation_only_timing_summary.json"
$summary | ConvertTo-Json -Depth 6 | Set-Content -LiteralPath $outJson -Encoding utf8
Write-Host "Wrote $outJson"
Get-Content $outJson
