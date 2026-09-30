$paths = @(
    'I:\Github\Latent_Style\WEAVE\runs\submission\canonical_root_epoch6\summary.json',
    'I:\Github\Latent_Style\WEAVE\runs\submission\legacy_repeat_epoch6\summary.json',
    'I:\Github\Latent_Style\WEAVE\exp\repro\fp32\summary.json',
    'I:\Github\Latent_Style\WEAVE\exp\repro\bf16\summary.json',
    'I:\Github\Latent_Style\WEAVE\exp\repro\bf16_adain20\summary.json',
    'I:\Github\Latent_Style\WEAVE\_tmp_opt_baseline\summary.json'
)
foreach ($p in $paths) {
    Write-Host "=== $p ==="
    if (Test-Path $p) {
        try {
            $j = Get-Content $p -Raw | ConvertFrom-Json
            Write-Host ("  checkpoint = " + $j.checkpoint)
            $a = $j.analysis
            if ($a.all_pairs_overview) {
                Write-Host "  [all_pairs_overview] keys:" ($a.all_pairs_overview.PSObject.Properties.Name -join ", ")
                $ov = $a.all_pairs_overview
                foreach ($k in @('clip_style','content_lpips','lpips','clip_s','clip_dir','clip_content','n_pairs','n_generated')) {
                    if ($ov.PSObject.Properties.Match($k).Count -gt 0) {
                        Write-Host ("    $k = " + $ov.$k)
                    }
                }
            }
            if ($a.style_transfer_ability) {
                Write-Host "  [style_transfer_ability] keys:" ($a.style_transfer_ability.PSObject.Properties.Name -join ", ")
            }
            if ($a.identity_reconstruction) {
                Write-Host "  [identity_reconstruction] keys:" ($a.identity_reconstruction.PSObject.Properties.Name -join ", ")
            }
            # idt_baselines
            if ($j.idt_baselines) {
                Write-Host "  [idt_baselines] keys:" ($j.idt_baselines.PSObject.Properties.Name -join ", ")
                $ib = $j.idt_baselines
                foreach ($k in @('clip_style','content_lpips','lpips','clip_s')) {
                    if ($ib.PSObject.Properties.Match($k).Count -gt 0) {
                        Write-Host ("    $k = " + $ib.$k)
                    }
                }
            }
        } catch {
            Write-Host "  parse error: $_"
        }
    } else {
        Write-Host "  (file not found)"
    }
    Write-Host ""
}
