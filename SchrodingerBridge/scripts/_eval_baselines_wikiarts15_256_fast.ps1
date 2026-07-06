# Fast evaluation of Identity / AdaIN / WCT on wikiarts-15 at 256 resolution (no summary_grid)
# 256 test set is at I:\datasets\wikiarts15_256_test (already created)
# Generated images need to be created fresh at 256.

$REPO = "I:\Github\Latent_Style\SchrodingerBridge"
$PYTHON = "C:\Program Files\Python312\python.exe"
$SRC_DIR = "$REPO\src"
$SCRIPTS_DIR = "$REPO\scripts"
$TEST_DIR = "I:\datasets\wikiarts15_256_test"
$OUT_ROOT = "$REPO\exp\baseline_wikiarts15_256"
$LOG = "$REPO\logs\baseline_wikiarts15_256.log"

$USER_SITE = "C:\Users\Administrator\AppData\Roaming\Python\Python312\site-packages"
$env:PYTHONPATH = "$SRC_DIR;$USER_SITE"
$env:PYTHONUSERBASE = "C:\Users\Administrator\AppData\Roaming\Python"
$env:CUDA_VISIBLE_DEVICES = "0"
$env:HF_HOME = "$REPO\exp\eval_cache\hf"
$env:TRANSFORMERS_OFFLINE = "0"

$STYLES = "Abstract_Expressionism,Art_Nouveau_Modern,Baroque,Color_Field_Painting,Cubism,Expressionism,Fauvism,High_Renaissance,Mannerism_Late_Renaissance,Naive_Art_Primitivism,Northern_Renaissance,Pop_Art,Post_Impressionism,Romanticism,Symbolism"

Set-Location $SRC_DIR
$started = Get-Date
"" | Out-File -FilePath $LOG -Encoding utf8
Add-Content $LOG "=== wikiarts-15 256 baselines eval started at $started ==="
Add-Content $LOG "  test_dir:   $TEST_DIR"
Add-Content $LOG "  out_root:   $OUT_ROOT"
Add-Content $LOG "  styles:     $STYLES"

# Step 1: Generate baseline images at 256
Add-Content $LOG "--- Step 1: Generate baseline images (256) ---"
$gen_methods = @("identity", "adain", "wct")
foreach ($m in $gen_methods) {
    $done = "$OUT_ROOT\$m\_DONE"
    if (Test-Path $done) {
        Add-Content $LOG "  [$m] _DONE marker exists, skipping generation."
        continue
    }
    $t0 = Get-Date
    Add-Content $LOG "  [$m] Generating at $t0..."
    $cmd_args = @(
        "-u", "$SCRIPTS_DIR\gen_trainfree_wikiarts15.py",
        "--method", "$m",
        "--image-root", "$TEST_DIR",
        "--output-root", "$OUT_ROOT",
        "--styles", "$STYLES",
        "--image-size", "256",
        "--max-src-per-style", "30"
    )
    & $PYTHON @cmd_args 2>&1 | Out-File -FilePath "$LOG.$m.gen" -Encoding utf8 -Append
    $ec = $LASTEXITCODE
    $t1 = Get-Date
    $dur = ($t1 - $t0).TotalSeconds
    if (Test-Path $done) {
        Add-Content $LOG "  [$m] GEN OK exit=$ec dur=${dur}s at $t1"
    } else {
        Add-Content $LOG "  [$m] GEN FAIL exit=$ec dur=${dur}s at $t1 (may still have images; check _DONE)"
    }
}

# Step 2: Evaluate each method (CLIP-S + LPIPS) without summary_grid
Add-Content $LOG "--- Step 2: Evaluate each method (CLIP-S + LPIPS, 256) ---"
$eval_methods = @("identity", "adain", "wct")
foreach ($m in $eval_methods) {
    $eval_dir = "$OUT_ROOT\$m"
    $csv_path = "$eval_dir\metrics.csv"
    if (Test-Path $csv_path) {
        Add-Content $LOG "  [$m] metrics.csv already exists, skipping eval."
        continue
    }
    $t0 = Get-Date
    Add-Content $LOG "  [$m] Evaluating at $t0..."
    $cmd_args = @(
        "-u", "utils\run_evaluation.py",
        "$eval_dir",
        "--reuse_generated",
        "--style_subdirs", "$STYLES",
        "--test_dir", "$TEST_DIR",
        "--eval_only_lpips_clip_style",
        "--max_src_samples", "30",
        "--batch_size", "2",
        "--ref_feature_batch_size", "2",
        "--no-save_summary_grid"
    )
    & $PYTHON @cmd_args 2>&1 | Out-File -FilePath "$LOG.$m.eval" -Encoding utf8 -Append
    $ec = $LASTEXITCODE
    $t1 = Get-Date
    $dur = ($t1 - $t0).TotalSeconds
    if ($ec -eq 0 -and (Test-Path $csv_path)) {
        Add-Content $LOG "  [$m] EVAL OK exit=$ec dur=${dur}s at $t1"
    } else {
        Add-Content $LOG "  [$m] EVAL FAIL exit=$ec dur=${dur}s at $t1"
    }
}

$ended = Get-Date
Add-Content $LOG "=== wikiarts-15 256 baselines eval finished at $ended (total $(($ended - $started).TotalSeconds)s) ==="
