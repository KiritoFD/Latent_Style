$ErrorActionPreference = 'SilentlyContinue'
Set-Location 'g:\GitHub\Latent_Style\SchrodingerBridge'

$logFile = '.trae\autoresearch\cleanup\logs\m5_cleanup.log'

function Remove-DirLogged {
    param([string]$Path, [string]$Reason)
    if (Test-Path $Path) {
        $items = Get-ChildItem $Path -Recurse -File -ErrorAction SilentlyContinue
        $size = 0
        if ($items) { $size = ($items | Measure-Object -Property Length -Sum).Sum }
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
        Add-Content -Path $logFile -Value "[$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')] SKIP $Path (not found) | $Reason"
        return 0
    }
}

$totalFreed = 0

Write-Host "=== Phase 8: exp/ smoke/probe (28 dirs) ==="
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

Write-Host "`n=== Phase 9: exp/ top-level loose files ==="
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

# Also delete wikiart_stress1 (if exists, with proper handling)
$stressDir = Get-ChildItem -Path 'exp' -Directory -Filter 'wikiart_stress1_*' -ErrorAction SilentlyContinue
if ($stressDir) {
    foreach ($d in $stressDir) {
        $totalFreed += Remove-DirLogged $d.FullName 'old series deprecated'
    }
} else {
    Write-Host "wikiart_stress1_* not found, skipping"
}

Write-Host "`n=== Phase 10: logs/ cleanup (eval image dirs misnamed as .txt) ==="
# logs/phase4e_db2_lvl1_result.txt/ contains evaluation images (80.7MB)
# logs/phase4g1a_train.txt/ and phase4g1b_train.txt/ also appear to be eval dirs
$txtDirs = @(
    'logs\phase4e_db2_lvl1_result.txt',
    'logs\phase4g1a_train.txt',
    'logs\phase4g1b_train.txt'
)
foreach ($d in $txtDirs) {
    if (Test-Path $d) {
        $items = Get-ChildItem $d -Recurse -File -ErrorAction SilentlyContinue
        $size = 0
        if ($items) { $size = ($items | Measure-Object -Property Length -Sum).Sum }
        $sizeMB = [math]::Round($size/1MB, 1)
        Remove-Item -Path $d -Recurse -Force -ErrorAction SilentlyContinue
        $totalFreed += $size
        Write-Host "Deleted $d ($sizeMB MB)"
        Add-Content -Path $logFile -Value "[$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')] DELETED $d | ${sizeMB}MB | logs/ misnamed eval image dir"
    }
}

Write-Host "`n=== Summary ==="
$totalGB = [math]::Round($totalFreed/1GB, 2)
Write-Host "Phase 8-10 freed: $totalGB GB"
Add-Content -Path $logFile -Value "`n=== Phase 8-10 freed: $totalGB GB ==="
