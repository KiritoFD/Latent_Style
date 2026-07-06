# Train WD-VF on WikiArt-15-256 (15 styles, 512 distinct5 disjoint)
# Dataset: I:/datasets/wikiarts15_256_test (test only, need training latents)
# Output: I:/Github/Latent_Style/SchrodingerBridge/exp/wikiarts15_256_wdvf/
#
# Steps:
#   1. Train WD-VF (resolution=256, 15 styles)
#   2. Evaluate (CLIP-S + LPIPS) on test set
#   3. Extract metrics and write to CSV
#
# This is Batch 2: WD-VF WikiArt-15-256 (~3 min training)

$REPO = "I:\Github\Latent_Style\SchrodingerBridge"
$PYTHON = "C:\Program Files\Python312\python.exe"
$SRC_DIR = "$REPO\src"
$SCRIPTS_DIR = "$REPO\scripts"
$LOG = "$REPO\logs\wikiarts15_256_wdvf_train.log"
$OUT_ROOT = "$REPO\exp\wikiarts15_256_wdvf"
$EVAL_OUT = "$OUT_ROOT\eval"

$USER_SITE = "C:\Users\Administrator\AppData\Roaming\Python\Python312\site-packages"
$env:PYTHONPATH = "$SRC_DIR;$USER_SITE"
$env:PYTHONUSERBASE = "C:\Users\Administrator\AppData\Roaming\Python"
$env:CUDA_VISIBLE_DEVICES = "0"
$env:HF_HOME = "$REPO\exp\eval_cache\hf"
$env:TRANSFORMERS_OFFLINE = "0"

# Dataset paths
$DATA_ROOT = "I:\datasets"
$TRAIN_DIR = "$DATA_ROOT\wikiarts15_256_512_train"
$TEST_DIR = "$DATA_ROOT\wikiarts15_256_test"

# Create output directories
New-Item -ItemType Directory -Force -Path "$REPO\logs" | Out-Null
New-Item -ItemType Directory -Force -Path $OUT_ROOT | Out-Null
New-Item -ItemType Directory -Force -Path $EVAL_OUT | Out-Null

"=== WD-VF WikiArt-15-256 training started at $(Get-Date) ===" | Tee-Object -FilePath $LOG -Append
"  test_dir:   $TEST_DIR" | Tee-Object -FilePath $LOG -Append
"  out_root:   $OUT_ROOT" | Tee-Object -FilePath $LOG -Append
"  output to:  $EVAL_OUT" | Tee-Object -FilePath $LOG -Append

# Check if training data exists
if (-not (Test-Path $TRAIN_DIR)) {
    "  WARNING: Training directory $TRAIN_DIR does not exist!" | Tee-Object -FilePath $LOG -Append
    "  Only test set exists. Will use test set for evaluation only (but we need training)." | Tee-Object -FilePath $LOG -Append
    exit 1
}

# Step 1: Train
"--- Step 1: Training WD-VF ---" | Tee-Object -FilePath $LOG -Append

$trainArgs = @(
    "-u",
    "$SRC_DIR\train.py",
    "--config", "$SCRIPTS_DIR\config_wikiarts15_256.json",
    "--output", "$OUT_ROOT"
)

"    CMD: $PYTHON $($trainArgs -join ' ')" | Tee-Object -FilePath $LOG -Append

try {
    $proc = Start-Process -FilePath $PYTHON -ArgumentList $trainArgs -NoNewWindow -PassThru -WorkingDirectory $REPO -RedirectStandardOutput "$LOG.train.out" -RedirectStandardError "$LOG.train.err"
    $proc.WaitForExit()
    $EXIT_CODE = $proc.ExitCode
    if ($EXIT_CODE -eq 0) {
        "  TRAIN SUCCESS at $(Get-Date)" | Tee-Object -FilePath $LOG -Append
    } else {
        "  TRAIN FAIL exit=$EXIT_CODE at $(Get-Date)" | Tee-Object -FilePath $LOG -Append
        "    last 20 lines of stderr:" | Tee-Object -FilePath $LOG -Append
        if (Test-Path "$LOG.train.err") {
            Get-Content "$LOG.train.err" -Tail 20 | ForEach-Object { "      $_" } | Tee-Object -FilePath $LOG -Append
        }
        exit 1
    }
} catch {
    "  TRAIN EXCEPTION: $_ at $(Get-Date)" | Tee-Object -FilePath $LOG -Append
    exit 1
}

# Step 2: Evaluate
"--- Step 2: Evaluate (CLIP-S + LPIPS) ---" | Tee-Object -FilePath $LOG -Append

# Find the best checkpoint (last epoch)
$ckpt = Get-ChildItem "$OUT_ROOT\checkpoints" -Filter "epoch_*.pt" | Sort-Object Name | Select-Object -Last 1
if (-not $ckpt) {
    "  EVAL FAIL: No checkpoint found in $OUT_ROOT\checkpoints" | Tee-Object -FilePath $LOG -Append
    exit 1
}
"  Found checkpoint: $($ckpt.Name)" | Tee-Object -FilePath $LOG -Append

$evalArgs = @(
    "-u",
    "$SRC_DIR\utils\run_evaluation.py",
    "--checkpoint", $ckpt.FullName,
    "--config", "$SCRIPTS_DIR\config_wikiarts15_256.json",
    "--test_dir", $TEST_DIR,
    "--output_dir", $EVAL_OUT,
    "--style_subdirs", "Abstract_Expressionism,Art_Nouveau_Modern,Baroque,Color_Field_Painting,Cubism,Expressionism,Fauvism,High_Renaissance,Mannerism_Late_Renaissance,Naive_Art_Primitivism,Northern_Renaissance,Pop_Art,Post_Impressionism,Romanticism,Symbolism",
    "--eval_only_lpips_clip_style",
    "--batch_size", "2",
    "--full_eval_batch_size", "2",
    "--ref_feature_batch_size", "2",
    "--max_src_samples", "30"
)

"    CMD: $PYTHON $($evalArgs -join ' ')" | Tee-Object -FilePath $LOG -Append

try {
    $proc = Start-Process -FilePath $PYTHON -ArgumentList $evalArgs -NoNewWindow -PassThru -WorkingDirectory $SRC_DIR -RedirectStandardOutput "$LOG.eval.out" -RedirectStandardError "$LOG.eval.err"
    $proc.WaitForExit()
    $EXIT_CODE = $proc.ExitCode
    if ($EXIT_CODE -eq 0) {
        "  EVAL SUCCESS at $(Get-Date)" | Tee-Object -FilePath $LOG -Append
    } else {
        "  EVAL FAIL exit=$EXIT_CODE at $(Get-Date)" | Tee-Object -FilePath $LOG -Append
        "    last 20 lines of stderr:" | Tee-Object -FilePath $LOG -Append
        if (Test-Path "$LOG.eval.err") {
            Get-Content "$LOG.eval.err" -Tail 20 | ForEach-Object { "      $_" } | Tee-Object -FilePath $LOG -Append
        }
        exit 1
    }
} catch {
    "  EVAL EXCEPTION: $_ at $(Get-Date)" | Tee-Object -FilePath $LOG -Append
    exit 1
}

"=== WD-VF WikiArt-15-256 finished at $(Get-Date) ===" | Tee-Object -FilePath $LOG -Append
exit 0
