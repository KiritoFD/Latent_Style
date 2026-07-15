$ErrorActionPreference = 'SilentlyContinue'
Set-Location 'g:\GitHub\Latent_Style\SchrodingerBridge'

$logFile = '.trae\autoresearch\cleanup\logs\m5_cleanup.log'
$ErrorActionPreference = 'Stop'

function Remove-DirLogged {
    param([string]$Path, [string]$Reason)
    if (Test-Path $Path) {
        $size = (Get-ChildItem $Path -Recurse -ErrorAction SilentlyContinue | Measure-Object -Property Length -Sum).Sum
        $sizeMB = [math]::Round($size/1MB, 1)
        $ts = Get-Date -Format 'yyyy-MM-dd HH:mm:ss'
        try {
            Remove-Item -Path $Path -Recurse -Force -ErrorAction Stop
            $entry = "[$ts] DELETED $Path | ${sizeMB}MB | $Reason"
            Write-Host $entry
            Add-Content -Path $logFile -Value $entry
            return $size
        } catch {
            $entry = "[$ts] FAILED $Path | ${sizeMB}MB | $Reason | $_"
            Write-Host $entry
            Add-Content -Path $logFile -Value $entry
            return 0
        }
    } else {
        $entry = "[$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')] SKIP $Path (not found) | $Reason"
        Add-Content -Path $logFile -Value $entry
        return 0
    }
}

# Ensure log dir exists
$logDir = Split-Path $logFile -Parent
if (-not (Test-Path $logDir)) { New-Item -ItemType Directory -Path $logDir -Force | Out-Null }
Set-Content -Path $logFile -Value "=== M5 Local Cleanup Log - $(Get-Date) ==="

$totalFreed = 0

Write-Host "`n=== Phase 1: Large unused directories ==="
$totalFreed += Remove-DirLogged 'scale' 'Old wikiart_1024 datasets from May 2026, no longer used (datasets now on /mnt/i/)'
$totalFreed += Remove-DirLogged '_codex_tmp' 'All temporary .log/.pid/.sh monitoring files'
$totalFreed += Remove-DirLogged 'aaai_submission_snapshot_9a4b99dfa_page1_scatter_artfid' 'Old snapshot, superseded'

Write-Host "`n=== Phase 2: aaai_submission tmp_* dirs ==="
Get-ChildItem -Path 'aaai_submission' -Directory -Filter 'tmp_*' | ForEach-Object {
    $totalFreed += Remove-DirLogged $_.FullName 'aaai_submission tmp_* render dir'
}
Get-ChildItem -Path 'aaai_submission' -Directory -Filter 'tmp_visual_*' | ForEach-Object {
    $totalFreed += Remove-DirLogged $_.FullName 'aaai_submission tmp_visual_* dir'
}

Write-Host "`n=== Phase 3: exp/ large old-series dirs ==="
$totalFreed += Remove-DirLogged 'exp\620_spatial_bridge' 'Old 620 series 65 smoke/ablation subdirs, superseded by 630 series'
$totalFreed += Remove-DirLogged 'exp\phase616_live_dashboard' 'Old 616 dashboard eval.tgz archives'

Write-Host "`n=== Phase 4: exp/ src_only (9 dirs) ==="
$srcOnly = @(
    '630_local_t11_long30ep',
    '630_local_t3_adain_ll_t3a',
    'task5_baseline_2ep',
    'task5_endpoint_a_2ep',
    'task5_endpoint_b_2ep',
    'task5_endpoint_c_2ep',
    'task6_baseline_5ep',
    'task6_exp_a_optimal_5ep',
    'task6_exp_b_two_stage_5ep'
)
foreach ($d in $srcOnly) {
    $totalFreed += Remove-DirLogged "exp\$d" 'src_only, no ckpt no eval'
}

Write-Host "`n=== Phase 5: exp/ temp script dirs (4 dirs) ==="
$tempDirs = @('625_fc_sb', 'p3_remote_10h', 'tuning_deepdive', 'phase4j_batch_logs')
foreach ($d in $tempDirs) {
    $totalFreed += Remove-DirLogged "exp\$d" 'temp script dir'
}

Write-Host "`n=== Phase 6: exp/ historical archive - May 2026 probes (36 dirs) ==="
$may2026 = @(
    'armored_breakthrough_proper',
    'decision_tree_clip_style',
    'diffeomorphic_tangent_head_sweep',
    'diffeomorphic_tangent_sweep',
    'fisher_operator_consumer_probe',
    'fisher_operator_tokenizer_probe',
    'fisher_style_backbone_probe',
    'fisher_style_memory_adapter_probe',
    'local_repro_sadd_38f_8ep_20260528_224707',
    'manual_k1_k2_8epoch',
    'phase1_diagnostic_probes',
    'physical_loss_tree',
    'reference_memory_generation_probe_full',
    'remote_factorized_tokenizer_pull',
    'router_aware_backbone_probe',
    'scripts',
    'style_memory_bank_adapter_probe',
    'style_memory_bank_adapter_route_probe',
    'style_memory_bank_probe',
    'style_memory_typed_adapter_probe',
    'style_representation_adapter_probe',
    'style_representation_safe_projection_probe',
    'style_representation_style_aware_router_probe',
    't01_local_base',
    'temp_anneal_proper',
    'tokenizer_adain_gate_calibration',
    'tokenizer_adain_texture_gate_calibration_rerun',
    'tokenizer_bandgate_calibration',
    'tokenizer_prototype_carrier_calibration',
    'tokenizer_stat_reader_probe',
    'tokenizer_stat_vocab_probe',
    'tokenizer_texton_carrier_calibration',
    'vae_backend_256_mse_controls',
    'vae_backend_256_status',
    'wikiart_512_encode_logs',
    'wikiart_512_transfer_logs'
)
foreach ($d in $may2026) {
    $totalFreed += Remove-DirLogged "exp\$d" 'historical May 2026 probe'
}

Write-Host "`n=== Phase 7: exp/ historical archive - old series (6 dirs) ==="
$oldSeries = @(
    'phase3_task1',
    'fc_sb_r2',
    '20250618_lite_ot_vertical_auto',
    'p4_fusion_breakout',
    'task4_no_dino'
)
foreach ($d in $oldSeries) {
    $totalFreed += Remove-DirLogged "exp\$d" 'old series deprecated'
}
# wikiart_stress1 long name
$stressDir = Get-ChildItem -Path 'exp' -Directory -Filter 'wikiart_stress1_*' -ErrorAction SilentlyContinue
foreach ($d in $stressDir) {
    $totalFreed += Remove-DirLogged $d.FullName 'old series deprecated'
}

Write-Host "`n=== Phase 8: exp/ smoke/probe (27 dirs) ==="
$smokeDirs = @(
    '_smoke_distinct5_512_ema_baseline_vlen004',
    '_smoke_distinct5_512_ema_variant_a_class_prototypes_b8_vlen001',
    '_smoke_distinct5_512_ema_variant_b_global_vq_b8_vlen001',
    '_smoke_distinct5_512_ema_variant_c_content_guided_spatial_b8_vlen001',
    '_smoke_distinct5_512_ema_variant_d_vq_content_guided_b8_vlen001',
    '_smoke_distinct5_512_ema_variant_e_latent_prototype_ot_queue_b8_vlen001',
    '_smoke_distinct5_512_ema_variant_i_dual_mix_local',
    '_smoke_distinct5_512_ema_variant_j_aux_hard_swd_local',
    '_smoke_distinct5_512_ema_variant_k_content_adaptive_local',
    '_smoke_distinct5_512_ema_variant_m_style_gated_local_windows',
    '_smoke_distinct5_profile_probe_b8_vlen001',
    '_smoke_distinct5_variant_a_latent_init',
    '_smoke_distinct5_variant_b_latent_init',
    'local_wsl_distinct5_512_ema_k_b16_step2min',
    'local_wsl_distinct5_512_ema_k_b16_step2min_v160',
    'local_wsl_distinct5_512_ema_k_b16_stepcalib',
    'local_wsl_distinct5_512_ema_k_b32_e8',
    'local_wsl_wikiart512_carrier_gate_from_hist_e3',
    'local_wsl_wikiart512_execution_budget_from_hist_e1',
    'local_wsl_wikiart512_full_b32_e8',
    'local_wsl_wikiart512_style_injection_delta_div_from_hist_e3',
    'local_wsl_wikiart512_style_injection_delta_div_w05_from_hist_e3',
    'local_wsl_wikiart512_style_injection_from_hist_e1',
    'local_wsl_wikiart512_style_injection_from_hist_e3',
    'probes_20260601',
    'smoke_blockmask',
    'style_representation_residual_scale_sweep',
    'tmp_output_appearance_resume_smoke'
)
foreach ($d in $smokeDirs) {
    $totalFreed += Remove-DirLogged "exp\$d" 'smoke/probe cleanup'
}

Write-Host "`n=== Phase 9: exp/ top-level loose files (logs, temp scripts) ==="
$looseFiles = @(
    'exp\630_local_t15_llgqca_train.log',
    'exp\630_local_t19b_dim96_train.log',
    'exp\630_phase3_train.log',
    'exp\630_phase4a2_adain_0_train.log',
    'exp\630_phase4a2_extrap_0_train.log',
    'exp\630_phase4b1_freq_a05_train.log',
    'exp\630_phase4b1_freq_a1_train.log',
    'exp\_eval_samam_unified.bat',
    'exp\_remote_scan_v2.py',
    'exp\ablation_log.md',
    'exp\analyze_task6_results.py',
    'exp\archive_manifest.json',
    'exp\baseline_err.log',
    'exp\baseline_train.log',
    'exp\clean_base_v2_relu2_train.log',
    'exp\gen_task6_configs.py',
    'exp\t5_err.log',
    'exp\t5_train.log',
    'exp\tuning_deepdive.zip',
    'exp\verify_task6_configs.py'
)
foreach ($f in $looseFiles) {
    if (Test-Path $f) {
        $size = (Get-Item $f).Length
        Remove-Item -Path $f -Force -ErrorAction SilentlyContinue
        $totalFreed += $size
        Add-Content -Path $logFile -Value "[$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')] DELETED $f | $([math]::Round($size/1MB,2))MB | loose file"
    }
}

Write-Host "`n=== Summary ==="
$totalGB = [math]::Round($totalFreed/1GB, 2)
Write-Host "Total freed: $totalGB GB"
Add-Content -Path $logFile -Value "`n=== Total freed: $totalGB GB ==="
