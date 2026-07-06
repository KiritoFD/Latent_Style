# Fast evaluation of AdaIN / WCT on wikiarts-15 (skip summary_grid to save time)
# Identity is skipped because its metrics.csv already exists from the prior run.

$REPO = "I:\Github\Latent_Style\SchrodingerBridge"
$PYTHON = "C:\Program Files\Python312\python.exe"
$SRC_DIR = "$REPO\src"
$TEST_DIR = "I:\datasets\wikiarts15_512_test"
$OUT_ROOT = "$REPO\exp\baseline_wikiarts15"
$LOG = "$REPO\logs\baseline_wikiarts15_fast.log"

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
Add-Content $LOG "=== wikiarts-15 fast eval (adain+wct, no summary_grid) started at $started ==="

# Methods to evaluate (identity already done)
$methods = @("adain", "wct")
foreach ($m in $methods) {
    $eval_dir = "$OUT_ROOT\$m"
    $csv_path = "$eval_dir\metrics.csv"
    if (Test-Path $csv_path) {
        Add-Content $Log "  [$m] metrics.csv already exists, skipping."
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
    & $PYTHON @cmd_args 2>&1 | Out-File -FilePath "$LOG.$m" -Encoding utf8 -Append
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
Add-Content $LOG "=== wikiarts-15 fast eval finished at $ended (total $(($ended - $started).TotalSeconds)s) ==="
