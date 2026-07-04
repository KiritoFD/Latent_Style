# Collect 628-ALL-DEBUG from X experiments - read tail lines via SSH, parse locally
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

$results = @()
foreach ($x in $xExps) {
    $logPath = "$remoteRoot\exp\628_ablation\destructive_logs\$($x.name).log"
    # Read last 200 lines via SSH (628-ALL-DEBUG every 100 steps, so last 200 lines should have step 3000+)
    $cmd = "powershell -NoProfile -Command Get-Content '$logPath' -Tail 200"
    $output = ssh -o ConnectTimeout=10 $remote -p $port $cmd 2>$null
    if (-not $output) {
        $results += [PSCustomObject]@{Name=$x.name; Loss=$x.loss; W=$x.w; Step='?'; LossVal='?'; Added='?'; FmTotal='?'; TotalLoss='?'; Status='NO_LOG'}
        continue
    }
    $lines = $output -split "`r?`n"
    $debugLines = $lines | Where-Object { $_ -match '628-ALL-DEBUG' }
    if ($debugLines.Count -eq 0) {
        $results += [PSCustomObject]@{Name=$x.name; Loss=$x.loss; W=$x.w; Step='?'; LossVal='?'; Added='?'; FmTotal='?'; TotalLoss='?'; Status='NO_DEBUG'}
        continue
    }
    $lastDebug = $debugLines[-1]
    $step = if ($lastDebug -match 'step=(\d+)') { $matches[1] } else { '?' }
    $fmTotal = if ($lastDebug -match 'fm_total=([\d.\-]+)') { $matches[1] } else { '?' }
    $totalLoss = if ($lastDebug -match 'total_loss=([\d.\-]+)') { $matches[1] } else { '?' }
    # Parse aux: lossname=val(w=W,added=A)
    $lossVal = if ($lastDebug -match 'aux: \w+=([\d.\-eE]+)') { $matches[1] } else { '?' }
    $added = if ($lastDebug -match 'added=([\d.\-eE]+)') { $matches[1] } else { '?' }
    $results += [PSCustomObject]@{Name=$x.name; Loss=$x.loss; W=$x.w; Step=$step; LossVal=$lossVal; Added=$added; FmTotal=$fmTotal; TotalLoss=$totalLoss; Status='OK'}
}

Write-Host "=== 628 X-Experiment Debug Summary (last 100-step output) ==="
Write-Host ""
$results | Format-Table -AutoSize
Write-Host ""
Write-Host "=== Grouped by Loss Type ==="
$results | Group-Object Loss | ForEach-Object {
    Write-Host ""
    Write-Host "--- $($_.Name) ---"
    $_.Group | Format-Table -AutoSize
}
