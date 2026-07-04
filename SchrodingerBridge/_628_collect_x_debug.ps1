# Collect 628-ALL-DEBUG step=3100 (last 100-step debug) from all X experiments
$remote = 'Administrator@100.115.18.62'
$port = '2222'
$remoteRoot = 'I:\Github\Latent_Style\SchrodingerBridge'

$xExps = @(
    @{name='X1_velmag_w10'; loss='vel_mag'; w=10},
    @{name='X2_velmag_w50'; loss='vel_mag'; w=50},
    @{name='X3_velmag_w100'; loss='vel_mag'; w=100},
    @{name='X4_dir_cos_w10'; loss='dir_cos'; w=10},
    @{name='X5_dir_cos_w50'; loss='dir_cos'; w=50},
    @{name='X6_dir_cos_w100'; loss='dir_cos'; w=100},
    @{name='X7_outvar_w10'; loss='out_var'; w=10},
    @{name='X8_outvar_w50'; loss='out_var'; w=50},
    @{name='X9_outvar_w100'; loss='out_var'; w=100},
    @{name='X10_contrast_w10'; loss='contrast'; w=10},
    @{name='X11_contrast_w50'; loss='contrast'; w=50},
    @{name='X12_contrast_w100'; loss='contrast'; w=100},
    @{name='X13_chvar_w10'; loss='ch_var'; w=10},
    @{name='X14_chvar_w50'; loss='ch_var'; w=50},
    @{name='X15_chvar_w100'; loss='ch_var'; w=100},
    @{name='X16_hfenergy_w10'; loss='hf_energy'; w=10},
    @{name='X17_hfenergy_w50'; loss='hf_energy'; w=50},
    @{name='X18_hfenergy_w100'; loss='hf_energy'; w=100},
    @{name='X19_colormatch_w10'; loss='color_match'; w=10},
    @{name='X20_colormatch_w50'; loss='color_match'; w=50},
    @{name='X21_colormatch_w100'; loss='color_match'; w=100},
    @{name='X22_hsvsat_w1'; loss='hsv_sat'; w=1},
    @{name='X23_hsvsat_w10'; loss='hsv_sat'; w=10},
    @{name='X24_hsvsat_w50'; loss='hsv_sat'; w=50},
    @{name='X25_attnent_w1'; loss='attn_ent'; w=1},
    @{name='X26_attnent_w10'; loss='attn_ent'; w=10},
    @{name='X27_attnent_w50'; loss='attn_ent'; w=50},
    @{name='X28_combo_content_w50'; loss='combo_content'; w=50},
    @{name='X29_combo_direction_w50'; loss='combo_direction'; w=50},
    @{name='X30_combo_all_w10'; loss='combo_all'; w=10},
    @{name='X31_combo_all_w50'; loss='combo_all'; w=50}
)

Write-Host "=== 628 X-Experiment Debug Summary (step=3100, last 100-step output) ==="
Write-Host ""
Write-Host "| Exp | Loss Type | Weight | Loss Value | Added | fm_total | total_loss |"
Write-Host "|-----|-----------|--------|------------|-------|----------|------------|"

foreach ($x in $xExps) {
    $logPath = "$remoteRoot\exp\628_ablation\destructive_logs\$($x.name).log"
    # Read last 50 lines and find the last 628-ALL-DEBUG line
    $cmd = "powershell -NoProfile -Command `"(Get-Content '$logPath' -Tail 50 | Select-String '628-ALL-DEBUG' | Select-Object -Last 1).Line`""
    $result = ssh -o ConnectTimeout=10 $remote -p $port $cmd 2>$null
    if ($result -and $result -match '628-ALL-DEBUG') {
        # Parse: [628-ALL-DEBUG] step=3100 fm_total=2.816123 total_loss=2.975678 aux_count=1 aux: contrast=0.015876(w=10.0,added=0.1588)
        $step = if ($result -match 'step=(\d+)') { $matches[1] } else { '?' }
        $fmTotal = if ($result -match 'fm_total=([\d.]+)') { $matches[1] } else { '?' }
        $totalLoss = if ($result -match 'total_loss=([\d.\-]+)') { $matches[1] } else { '?' }
        $lossVal = if ($result -match 'aux: \w+=([\d.\-]+)') { $matches[1] } else { '?' }
        $added = if ($result -match 'added=([\d.\-]+)') { $matches[1] } else { '?' }
        Write-Host "| $($x.name) | $($x.loss) | $($x.w) | $lossVal | $added | $fmTotal | $totalLoss |"
    } else {
        Write-Host "| $($x.name) | $($x.loss) | $($x.w) | NO DEBUG OUTPUT |"
    }
}
