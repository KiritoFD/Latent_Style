# Batch training script for 512 ablation v3 (48 configs) using Windows Python
# Supports eval-only mode: if checkpoint exists but summary.json missing, only run eval

$REPO = "I:\Github\Latent_Style\SchrodingerBridge"
$PYTHON = "C:\Program Files\Python312\python.exe"
$CONFIG_DIR = "$REPO\configs"
$EXP_ROOT = "$REPO\exp\abl512"
$LOG_DIR = "$REPO\logs"
$BATCH_LOG = "$LOG_DIR\abl512_v3_batch.log"
$SRC_DIR = "$REPO\src"

# Ensure directories exist
New-Item -ItemType Directory -Force -Path $LOG_DIR | Out-Null
New-Item -ItemType Directory -Force -Path $EXP_ROOT | Out-Null

# Set environment: include user site-packages (for SYSTEM account compatibility)
$USER_SITE = "C:\Users\Administrator\AppData\Roaming\Python\Python312\site-packages"
$env:PYTHONPATH = "$SRC_DIR;$USER_SITE"
$env:PYTHONUSERBASE = "C:\Users\Administrator\AppData\Roaming\Python"
$env:CUDA_VISIBLE_DEVICES = "0"
$env:HF_HOME = "$REPO\exp\eval_cache\hf"
$env:TRANSFORMERS_OFFLINE = "0"

# 48 experiment names
$EXPERIMENTS = @(
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

Set-Location $SRC_DIR

$TOTAL = $EXPERIMENTS.Count
$COUNT = 0
$SUCCESS = 0
$FAIL = 0
$SKIP = 0

$START_TIME = Get-Date
"========================================================" | Tee-Object -FilePath $BATCH_LOG -Append
"  abl512 v3 batch training started at $START_TIME" | Tee-Object -FilePath $BATCH_LOG -Append
"  Total experiments: $TOTAL" | Tee-Object -FilePath $BATCH_LOG -Append
"  Repo: $REPO" | Tee-Object -FilePath $BATCH_LOG -Append
"  Python: $PYTHON" | Tee-Object -FilePath $BATCH_LOG -Append
"  PYTHONPATH: $env:PYTHONPATH" | Tee-Object -FilePath $BATCH_LOG -Append
"========================================================" | Tee-Object -FilePath $BATCH_LOG -Append

foreach ($EXP in $EXPERIMENTS) {
    $COUNT++
    $CONFIG = "$CONFIG_DIR\abl512_$EXP.json"
    $LOG = "$LOG_DIR\abl512_v3_${EXP}_train.log"

    # Skip if final eval already exists
    $FINAL_EVAL = "$EXP_ROOT\$EXP\full_eval\epoch_0005\summary.json"
    if (Test-Path $FINAL_EVAL) {
        "[SKIP $COUNT/$TOTAL] $EXP - already has final eval" | Tee-Object -FilePath $BATCH_LOG -Append
        $SKIP++
        continue
    }

    # Verify config exists
    if (-not (Test-Path $CONFIG)) {
        "[MISS $COUNT/$TOTAL] $EXP - config not found: $CONFIG" | Tee-Object -FilePath $BATCH_LOG -Append
        $FAIL++
        continue
    }

    # Check if checkpoint exists but eval failed -> delete and retrain
    # (Training is fast ~2min, simpler than eval-only mode)
    $CKPT = "$EXP_ROOT\$EXP\epoch_0005.pt"
    if (-not (Test-Path $CKPT)) {
        $CKPT = "$EXP_ROOT\$EXP\epoch_0001.pt"  # For X45_epochs_1
    }
    if (Test-Path $CKPT) {
        # Checkpoint exists but no summary.json -> eval failed, delete and retrain
        "[RETRAIN $COUNT/$TOTAL] $EXP - checkpoint exists but eval failed, deleting and retraining" | Tee-Object -FilePath $BATCH_LOG -Append
        Remove-Item -Path "$EXP_ROOT\$EXP\epoch_*.pt" -Force -ErrorAction SilentlyContinue
        Remove-Item -Path "$EXP_ROOT\$EXP\full_eval" -Recurse -Force -ErrorAction SilentlyContinue
        Remove-Item -Path "$EXP_ROOT\$EXP\logs" -Recurse -Force -ErrorAction SilentlyContinue
    }

    $NOW = Get-Date
    "" | Tee-Object -FilePath $BATCH_LOG -Append
    "[START $COUNT/$TOTAL] $EXP at $NOW" | Tee-Object -FilePath $BATCH_LOG -Append
    "  config: $CONFIG" | Tee-Object -FilePath $BATCH_LOG -Append
    "  log:    $LOG" | Tee-Object -FilePath $BATCH_LOG -Append

    # Run training (capture output to per-exp log)
    # Use Start-Process to avoid PowerShell NativeCommandError on stderr (Python logging)
    try {
        $proc = Start-Process -FilePath $PYTHON -ArgumentList "-m", "run", "--config", $CONFIG -NoNewWindow -PassThru -WorkingDirectory $SRC_DIR -RedirectStandardOutput $LOG -RedirectStandardError "$LOG.err"
        $proc.WaitForExit()
        $EXIT_CODE = $proc.ExitCode
        if ($EXIT_CODE -eq 0) {
            "[DONE $COUNT/$TOTAL] $EXP - SUCCESS at $(Get-Date)" | Tee-Object -FilePath $BATCH_LOG -Append
            $SUCCESS++
        } else {
            "[FAIL $COUNT/$TOTAL] $EXP - exit code $EXIT_CODE at $(Get-Date)" | Tee-Object -FilePath $BATCH_LOG -Append
            "  last 10 lines of log:" | Tee-Object -FilePath $BATCH_LOG -Append
            if (Test-Path "$LOG.err") {
                Get-Content "$LOG.err" -Tail 10 | ForEach-Object { "    $_" } | Tee-Object -FilePath $BATCH_LOG -Append
            } elseif (Test-Path $LOG) {
                Get-Content $LOG -Tail 10 | ForEach-Object { "    $_" } | Tee-Object -FilePath $BATCH_LOG -Append
            }
            $FAIL++
        }
    } catch {
        "[FAIL $COUNT/$TOTAL] $EXP - exception: $_ at $(Get-Date)" | Tee-Object -FilePath $BATCH_LOG -Append
        $FAIL++
    }
}

$END_TIME = Get-Date
$DURATION = $END_TIME - $START_TIME
"" | Tee-Object -FilePath $BATCH_LOG -Append
"========================================================" | Tee-Object -FilePath $BATCH_LOG -Append
"  abl512 v3 batch training finished at $END_TIME" | Tee-Object -FilePath $BATCH_LOG -Append
"  Duration: $DURATION" | Tee-Object -FilePath $BATCH_LOG -Append
"  Total: $TOTAL | Success: $SUCCESS | Fail: $FAIL | Skip: $SKIP" | Tee-Object -FilePath $BATCH_LOG -Append
"========================================================" | Tee-Object -FilePath $BATCH_LOG -Append
