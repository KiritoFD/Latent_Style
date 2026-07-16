$p = "I:\Github\Latent_Style\WEAVE\_tmp_opt_v1\summary.json"
Write-Host "=== $p ==="
if (Test-Path $p) {
    $j = Get-Content $p -Raw | ConvertFrom-Json
    Write-Host ("  checkpoint     = " + $j.checkpoint)
    # settings
    $s = $j.settings
    Write-Host ("  batch_size     = " + $s.batch_size)
    Write-Host ("  gen_batch_size = " + $s.generation_batch_size)
    Write-Host ("  vae_compile    = " + $s.vae_compile_decoder)
    Write-Host ("  vae_compile_mode = " + $s.vae_compile_mode)
    Write-Host ("  vae_compile_fullgraph = " + $s.vae_compile_fullgraph)
    Write-Host ("  target_chunk   = " + $s.target_chunk_size)
    Write-Host ("  vae_decode_bs  = " + $s.vae_decode_batch_size)
    # timings
    $t = $j.timings_sec
    Write-Host ("  [TIMING]")
    Write-Host ("    wall_total       = " + $t.wall_total)
    Write-Host ("    lancet_generation= " + $t.lancet_generation)
    Write-Host ("    vae_decode       = " + $t.vae_decode)
    Write-Host ("    uint8_cpu_copy   = " + $t.uint8_cpu_copy)
    Write-Host ("    eval_total       = " + $t.eval_total)
    if ($t.image_save_join) { Write-Host ("    image_save_join  = " + $t.image_save_join) }
    if ($t.image_save_submit) { Write-Host ("    image_save_submit= " + $t.image_save_submit) }
    # analysis
    $a = $j.analysis.all_pairs_overview
    Write-Host ("  [METRICS]")
    Write-Host ("    clip_style       = " + $a.clip_style)
    Write-Host ("    content_lpips    = " + $a.content_lpips)
} else {
    Write-Host "  (file not found)"
}
