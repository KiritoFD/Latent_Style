# Monitor ablation progress: wait + summary
param([int]$WaitSec = 600)

Write-Host "Waiting $WaitSec seconds before checking progress..."
Start-Sleep -Seconds $WaitSec

$EXP_ROOT = "I:\Github\Latent_Style\SchrodingerBridge\exp\abl512"
$LOG = "I:\Github\Latent_Style\SchrodingerBridge\logs\abl512_v3_batch.log"

$EXPS = @(
    "X01_euler","X02_rk4","X03_steps_1","X04_steps_32","X05_corrector_4",
    "X06_no_spectral_ode","X07_spectral_levels_4","X08_spectral_levels_5",
    "X09_lowpass_avg","X10_w_ll_0","X11_w_hh_3x",
    "X12_adain_0","X13_adain_4x","X14_adain_every_step","X15_lowpass_1",
    "X16_lowpass_5","X17_velocity_floor_0","X18_velocity_floor_0p3",
    "X19_path_linear","X20_path_slerp","X21_sigma_0","X22_sigma_0p5","X23_no_target_proj",
    "X24_hungarian","X25_no_structure_cost","X26_structure_5x",
    "X27_sinkhorn_eps_0p5","X28_sinkhorn_iters_10",
    "X29_no_content_loss","X30_content_5x","X31_no_style_loss",
    "X32_style_32x","X33_style_64x","X34_no_flow","X35_no_kinetic",
    "X36_attn_softmax","X37_heads_1","X38_heads_16","X39_no_shortcut","X40_extrap_1",
    "X41_dim_32","X42_dim_128","X43_res_blocks_2","X44_no_skip",
    "X45_epochs_1","X46_lr_10x","X47_lr_0p1x","X48_t_uniform"
)

$DONE = 0
$EVAL_FAIL = 0
$TRAINING = 0
$NOT_STARTED = 0
$CURRENT_EXP = ""

foreach ($EXP in $EXPS) {
    $EVAL = "$EXP_ROOT\$EXP\full_eval\epoch_0005\summary.json"
    $EVAL1 = "$EXP_ROOT\$EXP\full_eval\epoch_0001\summary.json"
    $CKPT5 = "$EXP_ROOT\$EXP\epoch_0005.pt"
    $CKPT1 = "$EXP_ROOT\$EXP\epoch_0001.pt"
    if ((Test-Path $EVAL) -or (Test-Path $EVAL1)) {
        $DONE++
    } elseif ((Test-Path $CKPT5) -or (Test-Path $CKPT1)) {
        $EVAL_FAIL++
        if (-not $CURRENT_EXP) { $CURRENT_EXP = $EXP }
    } elseif (Test-Path "$EXP_ROOT\$EXP") {
        $TRAINING++
        if (-not $CURRENT_EXP) { $CURRENT_EXP = $EXP }
    } else {
        $NOT_STARTED++
    }
}

Write-Host ""
Write-Host "=== Progress Summary ==="
Write-Host "DONE (eval ok):       $DONE / $($EXPS.Count)"
Write-Host "EVAL_FAIL (ckpt only): $EVAL_FAIL"
Write-Host "TRAINING/FAILED:      $TRAINING"
Write-Host "NOT_STARTED:          $NOT_STARTED"
Write-Host "Current/Pending exp:  $CURRENT_EXP"

Write-Host ""
Write-Host "=== Batch log (last 5 lines) ==="
Get-Content $LOG -Tail 5

Write-Host ""
Write-Host "=== Python process ==="
Get-Process python -ErrorAction SilentlyContinue | Select-Object Id, CPU, StartTime | Format-Table

Write-Host ""
Write-Host "=== Latest err log (last 5 lines) ==="
$ERR_FILES = Get-ChildItem "I:\Github\Latent_Style\SchrodingerBridge\logs\abl512_v3_*_train.log.err" -ErrorAction SilentlyContinue | Sort-Object LastWriteTime -Descending | Select-Object -First 1
if ($ERR_FILES) {
    Write-Host "File: $($ERR_FILES.Name) (modified $($ERR_FILES.LastWriteTime))"
    Get-Content $ERR_FILES.FullName -Tail 5
}

# Show done experiment list with key metrics
Write-Host ""
Write-Host "=== Completed experiments with metrics ==="
foreach ($EXP in $EXPS) {
    $EVAL = "$EXP_ROOT\$EXP\full_eval\epoch_0005\summary.json"
    $EVAL1 = "$EXP_ROOT\$EXP\full_eval\epoch_0001\summary.json"
    $SUMMARY = $null
    if (Test-Path $EVAL) { $SUMMARY = Get-Content $EVAL | ConvertFrom-Json }
    elseif (Test-Path $EVAL1) { $SUMMARY = Get-Content $EVAL1 | ConvertFrom-Json }
    if ($SUMMARY) {
        $ts = $SUMMARY.analysis.style_transfer_ability.clip_style
        $tl = $SUMMARY.analysis.style_transfer_ability.content_lpips
        $as = $SUMMARY.analysis.all_pairs_overview.clip_style
        $al = $SUMMARY.analysis.all_pairs_overview.content_lpips
        Write-Host ("{0,-30} CLIP-S={1:F4} LPIPS={2:F4} | all-pairs: CLIP-S={3:F4} LPIPS={4:F4}" -f $EXP, $ts, $tl, $as, $al)
    }
}
