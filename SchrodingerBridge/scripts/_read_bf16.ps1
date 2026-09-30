$p = "C:\Users\Administrator\_tmp_bf16_eval\summary.json"
Write-Host "=== $p ==="
if (Test-Path $p) {
    $j = Get-Content $p -Raw | ConvertFrom-Json
    Write-Host ("  checkpoint     = " + $j.checkpoint)
    $s = $j.settings
    Write-Host ("  batch_size     = " + $s.batch_size)
    Write-Host ("  vae_compile    = " + $s.vae_compile_decoder)
    $t = $j.timings_sec
    Write-Host "[TIMING]"
    Write-Host ("  wall_total       = " + $t.wall_total)
    Write-Host ("  lancet_generation= " + $t.lancet_generation)
    Write-Host ("  vae_decode       = " + $t.vae_decode)
    Write-Host ("  uint8_cpu_copy   = " + $t.uint8_cpu_copy)
    Write-Host ("  eval_total       = " + $t.eval_total)
    $a = $j.analysis.all_pairs_overview
    Write-Host "[METRICS]"
    Write-Host ("  clip_style       = " + $a.clip_style)
    Write-Host ("  content_lpips    = " + $a.content_lpips)
} else {
    Write-Host "  (file not found)"
}