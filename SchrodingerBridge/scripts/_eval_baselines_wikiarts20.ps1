# Evaluate Identity / AdaIN / WCT baselines on wikiarts-20 (full 20 styles)
# Outputs to exp/baseline_wikiarts20/{method}/
#
# Pipeline per method:
#   1. Generate baseline images via scripts/gen_trainfree_wikiarts20.py
#      -> {OUT_ROOT}/{method}/images/*.png  +  {OUT_ROOT}/{method}/_DONE
#      (skip-resumable: existing PNGs are skipped, only missing distinct5 pairs generated)
#   2. Evaluate CLIP-S + LPIPS via src/utils/run_evaluation.py --reuse_generated
#      -> metrics.csv under {OUT_ROOT}/{method}/ (overwrites 15-style metrics)
#
# Methods are invoked as separate Python processes for fault isolation.

$REPO = "I:\Github\Latent_Style\SchrodingerBridge"
$PYTHON = "C:\Program Files\Python312\python.exe"
$SRC_DIR = "$REPO\src"
$SCRIPTS_DIR = "$REPO\scripts"
$TEST_DIR = "I:\datasets\wikiarts20_512_test"
$OUT_ROOT = "$REPO\exp\baseline_wikiarts20"
$LOG = "$REPO\logs\baseline_wikiarts20.log"

$USER_SITE = "C:\Users\Administrator\AppData\Roaming\Python\Python312\site-packages"
$env:PYTHONPATH = "$SRC_DIR;$USER_SITE"
$env:PYTHONUSERBASE = "C:\Users\Administrator\AppData\Roaming\Python"
$env:CUDA_VISIBLE_DEVICES = "0"
$env:HF_HOME = "$REPO\exp\eval_cache\hf"
$env:TRANSFORMERS_OFFLINE = "0"

# wikiarts-20: full 20 styles (15 non-distinct5 + 5 distinct5)
$STYLES = "Abstract_Expressionism,Art_Nouveau_Modern,Baroque,Color_Field_Painting,Cubism,Early_Renaissance,Expressionism,Fauvism,High_Renaissance,Impressionism,Mannerism_Late_Renaissance,Minimalism,Naive_Art_Primitivism,Northern_Renaissance,Pop_Art,Post_Impressionism,Rococo,Romanticism,Symbolism,Ukiyo_e"

# Methods to evaluate (order: identity first since it's CPU-only and fast)
$METHODS = @("identity", "adain", "wct")

New-Item -ItemType Directory -Force -Path "$REPO\logs" | Out-Null
New-Item -ItemType Directory -Force -Path $OUT_ROOT | Out-Null

"=== wikiarts-20 baselines eval started at $(Get-Date) ===" | Tee-Object -FilePath $LOG -Append
"  test_dir:   $TEST_DIR" | Tee-Object -FilePath $LOG -Append
"  out_root:   $OUT_ROOT" | Tee-Object -FilePath $LOG -Append
"  styles:     $STYLES" | Tee-Object -FilePath $LOG -Append
"  methods:    $($METHODS -join ', ')" | Tee-Object -FilePath $LOG -Append

# Remove old _DONE markers so generation runs again (skip-resumable will keep existing PNGs)
foreach ($method in $METHODS) {
    $doneMarker = "$OUT_ROOT\$method\_DONE"
    if (Test-Path $doneMarker) {
        Remove-Item $doneMarker -Force
        "  [$method] removed old _DONE marker to allow incremental generation" | Tee-Object -FilePath $LOG -Append
    }
    # Remove old 15-style metrics.csv so we get a fresh 20-style evaluation
    $oldCsv = "$OUT_ROOT\$method\metrics.csv"
    if (Test-Path $oldCsv) {
        Remove-Item $oldCsv -Force
        "  [$method] removed old 15-style metrics.csv" | Tee-Object -FilePath $LOG -Append
    }
}

# ── Step 1: Generate baseline images (per method, skip-resumable) ──
"--- Step 1: Generate baseline images (skip-resumable) ---" | Tee-Object -FilePath $LOG -Append

foreach ($method in $METHODS) {
    "  [$method] Generating at $(Get-Date)..." | Tee-Object -FilePath $LOG -Append

    $genArgs = @(
        "-u",
        "$SCRIPTS_DIR\gen_trainfree_wikiarts20.py",
        "--method", $method,
        "--image-root", $TEST_DIR,
        "--output-root", $OUT_ROOT,
        "--styles", $STYLES,
        "--image-size", "512",
        "--max-src-per-style", "30"
    )

    "    CMD: $PYTHON $($genArgs -join ' ')" | Tee-Object -FilePath $LOG -Append

    try {
        $proc = Start-Process -FilePath $PYTHON -ArgumentList $genArgs -NoNewWindow -PassThru -WorkingDirectory $REPO -RedirectStandardOutput "$LOG.$method.gen.out" -RedirectStandardError "$LOG.$method.gen.err"
        $proc.WaitForExit()
        $EXIT_CODE = $proc.ExitCode
        if ($EXIT_CODE -eq 0) {
            "  [$method] GEN SUCCESS at $(Get-Date)" | Tee-Object -FilePath $LOG -Append
            if (Test-Path "$LOG.$method.gen.out") {
                Get-Content "$LOG.$method.gen.out" -Tail 5 | ForEach-Object { "    $_" } | Tee-Object -FilePath $LOG -Append
            }
        } else {
            "  [$method] GEN FAIL exit=$EXIT_CODE at $(Get-Date)" | Tee-Object -FilePath $LOG -Append
            "    last 20 lines of stderr:" | Tee-Object -FilePath $LOG -Append
            if (Test-Path "$LOG.$method.gen.err") {
                Get-Content "$LOG.$method.gen.err" -Tail 20 | ForEach-Object { "      $_" } | Tee-Object -FilePath $LOG -Append
            }
        }
    } catch {
        "  [$method] GEN EXCEPTION: $_ at $(Get-Date)" | Tee-Object -FilePath $LOG -Append
    }
}

# ── Step 2: Evaluate each method (CLIP-S + LPIPS, 20 styles) ──
"--- Step 2: Evaluate each method (CLIP-S + LPIPS, 20 styles) ---" | Tee-Object -FilePath $LOG -Append

foreach ($method in $METHODS) {
    $methodDir = "$OUT_ROOT\$method"
    $doneMarker = "$methodDir\_DONE"

    if (-not (Test-Path $doneMarker)) {
        "  [$method] EVAL SKIP: _DONE marker not found (generation failed or incomplete)" | Tee-Object -FilePath $LOG -Append
        continue
    }

    $imgCount = (Get-ChildItem (Join-Path $methodDir "images") -Filter "*.png" -ErrorAction SilentlyContinue).Count
    "  [$method] image count: $imgCount (expected 12000 for 20x20x30)" | Tee-Object -FilePath $LOG -Append

    "  [$method] Evaluating at $(Get-Date)..." | Tee-Object -FilePath $LOG -Append

    # eval_dir = methodDir (contains images/*.png matching *_to_*.png pattern)
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
        $EXIT_CODE = $proc.ExitCode
        if ($EXIT_CODE -eq 0) {
            "  [$method] EVAL SUCCESS at $(Get-Date)" | Tee-Object -FilePath $LOG -Append
        } else {
            "  [$method] EVAL FAIL exit=$EXIT_CODE at $(Get-Date)" | Tee-Object -FilePath $LOG -Append
            "    last 20 lines of stderr:" | Tee-Object -FilePath $LOG -Append
            if (Test-Path "$LOG.$method.eval.err") {
                Get-Content "$LOG.$method.eval.err" -Tail 20 | ForEach-Object { "      $_" } | Tee-Object -FilePath $LOG -Append
            }
        }
    } catch {
        "  [$method] EVAL EXCEPTION: $_ at $(Get-Date)" | Tee-Object -FilePath $LOG -Append
    }
}

"=== wikiarts-20 baselines eval finished at $(Get-Date) ===" | Tee-Object -FilePath $LOG -Append
