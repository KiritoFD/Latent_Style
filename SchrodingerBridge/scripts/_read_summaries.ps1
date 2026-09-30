$paths = @(
    'I:\Github\Latent_Style\WEAVE\exp\repro\fp32\summary.json',
    'I:\Github\Latent_Style\WEAVE\exp\repro\bf16\summary.json',
    'I:\Github\Latent_Style\WEAVE\exp\repro\bf16_adain20\summary.json',
    'I:\Github\Latent_Style\WEAVE\runs\submission\canonical_root_epoch6\summary.json',
    'I:\Github\Latent_Style\WEAVE\runs\submission\legacy_repeat_epoch6\summary.json',
    'I:\Github\Latent_Style\WEAVE\_tmp_opt_baseline\summary.json'
)
foreach ($p in $paths) {
    Write-Host "=== $p ==="
    if (Test-Path $p) {
        try {
            $j = Get-Content $p -Raw | ConvertFrom-Json
            Write-Host ("  checkpoint     = " + $j.checkpoint)
            # settings
            if ($j.settings) {
                $s = $j.settings
                Write-Host ("  [settings] keys: " + ($s.PSObject.Properties.Name -join ", "))
                foreach ($k in @('endpoint_adain_scale','solver_family','batch_size','generation_batch_size','vae_compile_decoder','vae_compile_mode','vae_compile_fullgraph','target_chunk_size','num_steps','vae_decode_batch_size','metric_batch_size','vae_compile_method')) {
                    if ($s.PSObject.Properties.Match($k).Count -gt 0) {
                        Write-Host ("    $k = " + $s.$k)
                    }
                }
            }
            # timings_sec
            if ($j.timings_sec) {
                $t = $j.timings_sec
                Write-Host ("  [timings] keys: " + ($t.PSObject.Properties.Name -join ", "))
                foreach ($k in @('wall_total','lancet_generation','vae_decode','eval_total','uint8_cpu_copy','total')) {
                    if ($t.PSObject.Properties.Match($k).Count -gt 0) {
                        Write-Host ("    $k = " + $t.$k)
                    }
                }
            }
            # analysis (metrics)
            if ($j.analysis) {
                $a = $j.analysis
                Write-Host ("  [analysis] keys: " + ($a.PSObject.Properties.Name -join ", "))
                foreach ($k in @('clip_style','content_lpips','lpips','dino_style','dino_content','dino_s','dino_c','clip_s','art_fid')) {
                    if ($a.PSObject.Properties.Match($k).Count -gt 0) {
                        Write-Host ("    $k = " + $a.$k)
                    }
                }
            }
            # metrics_note
            if ($j.metrics_note) {
                Write-Host ("  metrics_note   = " + $j.metrics_note)
            }
        } catch {
            Write-Host "  parse error: $_"
        }
    } else {
        Write-Host "  (file not found)"
    }
    Write-Host ""
}
