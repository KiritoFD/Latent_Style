# Evaluate WD-VF (random-20 20-style model) on wikiarts-20 (full 20 styles)
# Outputs to exp/wikiarts20_eval/ (overwrites old 15-style metrics)

$REPO = "I:\Github\Latent_Style\SchrodingerBridge"
$PYTHON = "C:\Program Files\Python312\python.exe"
$SRC_DIR = "$REPO\src"
$CKPT = "$REPO\exp\wikiarts20_eval\epoch_0005.pt"
$CONFIG = "$REPO\exp\wikiarts20_eval\config.json"
$TEST_DIR = "I:\datasets\wikiarts20_512_test"
$OUT_DIR = "$REPO\exp\wikiarts20_eval"
$LOG = "$REPO\logs\wikiarts20_eval.log"

$USER_SITE = "C:\Users\Administrator\AppData\Roaming\Python\Python312\site-packages"
$env:PYTHONPATH = "$SRC_DIR;$USER_SITE"
$env:PYTHONUSERBASE = "C:\Users\Administrator\AppData\Roaming\Python"
$env:CUDA_VISIBLE_DEVICES = "0"
$env:HF_HOME = "$REPO\exp\eval_cache\hf"
$env:TRANSFORMERS_OFFLINE = "0"

# wikiarts-20: full 20 styles (15 non-distinct5 + 5 distinct5)
$STYLES = "Abstract_Expressionism,Art_Nouveau_Modern,Baroque,Color_Field_Painting,Cubism,Early_Renaissance,Expressionism,Fauvism,High_Renaissance,Impressionism,Mannerism_Late_Renaissance,Minimalism,Naive_Art_Primitivism,Northern_Renaissance,Pop_Art,Post_Impressionism,Rococo,Romanticism,Symbolism,Ukiyo_e"

New-Item -ItemType Directory -Force -Path "$REPO\logs" | Out-Null
New-Item -ItemType Directory -Force -Path $OUT_DIR | Out-Null

# Remove old 15-style metrics.csv so we get a fresh 20-style evaluation
$oldCsv = "$OUT_DIR\metrics.csv"
if (Test-Path $oldCsv) {
    Remove-Item $oldCsv -Force
    "  removed old 15-style metrics.csv" | Tee-Object -FilePath $LOG -Append
}
$oldImagesDir = "$OUT_DIR\images"
if (Test-Path $oldImagesDir) {
    Remove-Item $oldImagesDir -Recurse -Force
    "  removed old 15-style images dir" | Tee-Object -FilePath $LOG -Append
}

Set-Location $SRC_DIR

"=== wikiarts-20 WD-VF eval started at $(Get-Date) ===" | Tee-Object -FilePath $LOG -Append
"  checkpoint: $CKPT" | Tee-Object -FilePath $LOG -Append
"  test_dir:   $TEST_DIR" | Tee-Object -FilePath $LOG -Append
"  styles:     $STYLES" | Tee-Object -FilePath $LOG -Append

# Run evaluation
try {
    $proc = Start-Process -FilePath $PYTHON -ArgumentList "-u", "utils\run_evaluation.py", "--checkpoint", $CKPT, "--config", $CONFIG, "--test_dir", $TEST_DIR, "--style_subdirs", $STYLES, "--output", $OUT_DIR, "--eval_only_lpips_clip_style", "--max_src_samples", "30", "--batch_size", "2", "--ref_feature_batch_size", "2" -NoNewWindow -PassThru -WorkingDirectory $SRC_DIR -RedirectStandardOutput "$LOG.out" -RedirectStandardError "$LOG.err"
    $proc.WaitForExit()
    $EXIT_CODE = $proc.ExitCode
    if ($EXIT_CODE -eq 0) {
        "=== SUCCESS at $(Get-Date) ===" | Tee-Object -FilePath $LOG -Append
    } else {
        "=== FAIL exit=$EXIT_CODE at $(Get-Date) ===" | Tee-Object -FilePath $LOG -Append
        "  last 20 lines of stderr:" | Tee-Object -FilePath $LOG -Append
        if (Test-Path "$LOG.err") {
            Get-Content "$LOG.err" -Tail 20 | ForEach-Object { "    $_" } | Tee-Object -FilePath $LOG -Append
        }
    }
} catch {
    "=== EXCEPTION: $_ at $(Get-Date) ===" | Tee-Object -FilePath $LOG -Append
}
