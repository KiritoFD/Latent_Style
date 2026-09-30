# Compare epoch_0004 DINO metrics with main table
$paths = @(
    'I:\Github\Latent_Style\WEAVE\runs\submission\repro_brk_a_15ep\paper_eval_adain20\epoch_0004\dino_summary.json',
    'I:\Github\Latent_Style\WEAVE\runs\submission\repro_brk_a_15ep\paper_eval_adain20\epoch_0004\summary.json'
)

Write-Host "=== Main table targets (paper.tex) ==="
Write-Host "  DINO-S = 0.4915"
Write-Host "  CLIP-S = 0.7126"
Write-Host "  LPIPS  = 0.2596"
Write-Host "  DINO-C = 0.8103"
Write-Host ""

Write-Host "=== epoch_0004 DINO summary ==="
if (Test-Path $paths[0]) {
    $j = Get-Content $paths[0] -Raw | ConvertFrom-Json
    Write-Host ("  protocol       = " + $j.protocol)
    Write-Host ("  n_all          = " + $j.n_all)
    Write-Host ("  n_off_diagonal = " + $j.n_off_diagonal)
    Write-Host ("  all_dino_s     = " + $j.all_dino_s)
    Write-Host ("  all_dino_c     = " + $j.all_dino_c)
    Write-Host ("  all_dino_struct= " + $j.all_dino_structure)
    Write-Host ("  off_dino_s     = " + $j.off_dino_s)
    Write-Host ("  off_dino_c     = " + $j.off_dino_c)
    Write-Host ("  all_clip_s     = " + $j.all_clip_s)
    Write-Host ("  all_lpips      = " + $j.all_lpips)
}

Write-Host ""
Write-Host "=== epoch_0004 CLIP/LPIPS summary (timing) ==="
if (Test-Path $paths[1]) {
    $j = Get-Content $paths[1] -Raw | ConvertFrom-Json
    Write-Host ("  checkpoint     = " + $j.checkpoint)
    Write-Host ("  wall_total     = " + $j.timings_sec.wall_total)
    Write-Host ("  lancet_gen     = " + $j.timings_sec.lancet_generation)
    Write-Host ("  vae_decode     = " + $j.timings_sec.vae_decode)
    Write-Host ("  uint8_cpu_copy = " + $j.timings_sec.uint8_cpu_copy)
    Write-Host ("  eval_total     = " + $j.timings_sec.eval_total)
    $a = $j.analysis.all_pairs_overview
    Write-Host ("  clip_style     = " + $a.clip_style)
    Write-Host ("  content_lpips  = " + $a.content_lpips)
    $s = $j.settings
    Write-Host ("  batch_size     = " + $s.batch_size)
    Write-Host ("  gen_batch_size = " + $s.generation_batch_size)
    Write-Host ("  vae_compile    = " + $s.vae_compile_decoder)
    Write-Host ("  target_chunk   = " + $s.target_chunk_size)
}

Write-Host ""
Write-Host "=== GPU info ==="
& "C:\Program Files\Python312\python.exe" -c "import torch; print('torch:', torch.__version__); print('cuda:', torch.cuda.is_available()); print('device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A'); print('SM count:', torch.cuda.get_device_properties(0).multi_processor_count if torch.cuda.is_available() else 'N/A')" 2>&1
