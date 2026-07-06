# Evaluate Identity / AdaIN / WCT baselines on wikiarts-15 at 256 resolution
# Outputs to exp/baseline_wikiarts15_256/{method}/
#
# Uses the 256-resolution test set at I:\datasets\wikiarts15_256_test

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

# wikiarts-15: 15 styles (random-20 minus distinct5)
$STYLES = "Abstract_Expressionism,Art_Nouveau_Modern,Baroque,Color_Field_Painting,Cubism,Expressionism,Fauvism,High_Renaissance,Mannerism_Late_Renaissance,Naive_Art_Primitivism,Northern_Renaissance,Pop_Art,Post_Impressionism,Romanticism,Symbolism"

$METHODS = @("identity", "adain", "wct")

New-Item -ItemType Directory -Force -Path "$REPO\logs" | Out-Null
New-Item -ItemType Directory -Force -Path $OUT_ROOT | Out-Null

"=== wikiarts-15 256 baselines eval started at $(Get-Date) ===" | Tee-Object -FilePath $LOG -Append
"  test_dir:   $TEST_DIR" | Tee-Object -FilePath $LOG -Append
"  out_root:   $OUT_ROOT" | Tee-Object -FilePath $LOG -Append
"  styles:     $STYLES" | Tee-Object -FilePath $LOG -Append
"  methods:    $($METHODS -join ', ')" | Tee-Object -FilePath $LOG -Append

# Step 1: Generate baseline images (per method, skip if _DONE exists)
"--- Step 1: Generate baseline images (256) ---" | Tee-Object -FilePath $LOG -Append

foreach ($method in $METHODS) {
    $doneMarker = "$OUT_ROOT\$method\_DONE"
    if (Test-Path $doneMarker) {
        "  [$method] _DONE marker found, skipping generation" | Tee-Object -FilePath $LOG -Append
        continue
    }

    "  [$method] Generating at $(Get-Date)..." | Tee-Object -FilePath $LOG -Append

    $genArgs = @(
        "-u",
        "$SCRIPTS_DIR\gen_trainfree_wikiarts15.py",
        "--method", $method,
        "--image-root", $TEST_DIR,
        "--output-root", $OUT_ROOT,
        "--styles", $STYLES,
        "--image-size", "256",
        "--max-src-per-style", "30"
    )

    "    CMD: $PYTHON $($genArgs -join ' ')" | Tee-Object -FilePath $LOG -Append

    try {
        $proc = Start-Process -FilePath $PYTHON -ArgumentList $genArgs -NoNewWindow -PassThru -WorkingDirectory $REPO -RedirectStandardOutput "$LOG.$method.gen.out" -RedirectStandardError "$LOG.$method.gen.err"
        $proc.WaitForExit()
        if (Test-Path $doneMarker) {
            "  [$method] GEN SUCCESS at $(Get-Date)" | Tee-Object -FilePath $LOG -Append
        } else {
            "  [$method] GEN may have failed (no _DONE marker) at $(Get-Date)" | Tee-Object -FilePath $LOG -Append
            if (Test-Path "$LOG.$method.gen.err") {
                Get-Content "$LOG.$method.gen.err" -Tail 20 | ForEach-Object { "      $_" } | Tee-Object -FilePath $LOG -Append
            }
        }
    } catch {
        "  [$method] GEN EXCEPTION: $_ at $(Get-Date)" | Tee-Object -FilePath $LOG -Append
    }
}

# Step 2: Evaluate each method (CLIP-S + LPIPS)
"--- Step 2: Evaluate each method (CLIP-S + LPIPS) ---" | Tee-Object -FilePath $LOG -Append

foreach ($method in $METHODS) {
    $methodDir = "$OUT_ROOT\$method"
    $doneMarker = "$methodDir\_DONE"

    if (-not (Test-Path $doneMarker)) {
        "  [$method] EVAL SKIP: _DONE marker not found" | Tee-Object -FilePath $LOG -Append
        continue
    }

    "  [$method] Evaluating at $(Get-Date)..." | Tee-Object -FilePath $LOG -Append

    $evalArgs = @(
        "-u",
        "utils\run_evaluation.py",
        $methodDir,
        "--reuse_generated",
        "--style_subdirs", $STYLES,
        "--test_dir", $TEST_DIR,
        "--eval_only_lpips_clip_style",
        "--max_src_samples", "30",
        "--batch_size", "2",
        "--ref_feature_batch_size", "2"
    )

    "    CMD: $PYTHON $($evalArgs -join ' ')" | Tee-Object -FilePath $LOG -Append

    try {
        $proc = Start-Process -FilePath $PYTHON -ArgumentList $evalArgs -NoNewWindow -PassThru -WorkingDirectory $SRC_DIR -RedirectStandardOutput "$LOG.$method.eval.out" -RedirectStandardError "$LOG.$method.eval.err"
        $proc.WaitForExit()
        "  [$method] EVAL done at $(Get-Date)" | Tee-Object -FilePath $LOG -Append
    } catch {
        "  [$method] EVAL EXCEPTION: $_ at $(Get-Date)" | Tee-Object -FilePath $LOG -Append
    }
}

"=== wikiarts-15 256 baselines eval finished at $(Get-Date) ===" | Tee-Object -FilePath $LOG -Append
