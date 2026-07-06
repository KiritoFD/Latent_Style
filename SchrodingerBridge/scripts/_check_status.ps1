# Check ablation training status
$LOG = "I:\Github\Latent_Style\SchrodingerBridge\logs\abl512_v3_batch.log"
$EXP_ROOT = "I:\Github\Latent_Style\SchrodingerBridge\exp\abl512"

Write-Host "=== Batch log (last 30 lines) ==="
if (Test-Path $LOG) {
    Get-Content $LOG -Tail 30
} else {
    Write-Host "Batch log not found"
}

Write-Host ""
Write-Host "=== Python processes ==="
Get-Process python -ErrorAction SilentlyContinue | Select-Object Id, CPU, WorkingSet, StartTime | Format-Table

Write-Host ""
Write-Host "=== Experiment status ==="
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

$SUCCESS = 0
$FAIL = 0
$PENDING = 0
$MISSING = 0
foreach ($EXP in $EXPS) {
    $EVAL = "$EXP_ROOT\$EXP\full_eval\epoch_0005\summary.json"
    $EVAL1 = "$EXP_ROOT\$EXP\full_eval\epoch_0001\summary.json"
    $CKPT5 = "$EXP_ROOT\$EXP\epoch_0005.pt"
    $CKPT1 = "$EXP_ROOT\$EXP\epoch_0001.pt"
    if ((Test-Path $EVAL) -or (Test-Path $EVAL1)) {
        $SUCCESS++
        $status = "DONE"
    } elseif ((Test-Path $CKPT5) -or (Test-Path $CKPT1)) {
        $PENDING++
        $status = "EVAL_FAIL"
    } elseif (Test-Path "$EXP_ROOT\$EXP") {
        $PENDING++
        $status = "TRAINING/FAILED"
    } else {
        $MISSING++
        $status = "NOT_STARTED"
    }
    Write-Host ("{0,-30} {1}" -f $EXP, $status)
}
Write-Host ""
Write-Host "Summary: DONE=$SUCCESS, EVAL_FAIL=$PENDING, NOT_STARTED=$MISSING, Total=$($EXPS.Count)"

Write-Host ""
Write-Host "=== Latest err log (if any) ==="
$ERR_FILES = Get-ChildItem "I:\Github\Latent_Style\SchrodingerBridge\logs\abl512_v3_*_train.log.err" -ErrorAction SilentlyContinue | Sort-Object LastWriteTime -Descending | Select-Object -First 1
if ($ERR_FILES) {
    Write-Host "File: $($ERR_FILES.Name)"
    Get-Content $ERR_FILES.FullName -Tail 20
}
