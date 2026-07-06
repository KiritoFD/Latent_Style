# Evaluate WD-VF (random-20 20-style model) on wikiarts-15 (15 styles, distinct5 excluded)
# Outputs to exp/wikiarts15_eval/full_eval/

$REPO = "I:\Github\Latent_Style\SchrodingerBridge"
$PYTHON = "C:\Program Files\Python312\python.exe"
$SRC_DIR = "$REPO\src"
$CKPT = "$REPO\exp\wikiarts15_eval\epoch_0005.pt"
$CONFIG = "$REPO\exp\wikiarts15_eval\config.json"
$TEST_DIR = "I:\datasets\wikiarts15_512_test"
$OUT_DIR = "$REPO\exp\wikiarts15_eval"
$LOG = "$REPO\logs\wikiarts15_eval.log"

$USER_SITE = "C:\Users\Administrator\AppData\Roaming\Python\Python312\site-packages"
$env:PYTHONPATH = "$SRC_DIR;$USER_SITE"
$env:PYTHONUSERBASE = "C:\Users\Administrator\AppData\Roaming\Python"
$env:CUDA_VISIBLE_DEVICES = "0"
$env:HF_HOME = "$REPO\exp\eval_cache\hf"
$env:TRANSFORMERS_OFFLINE = "0"

# wikiarts-15: 15 styles (random-20 minus distinct5)
$STYLES = "Abstract_Expressionism,Art_Nouveau_Modern,Baroque,Color_Field_Painting,Cubism,Expressionism,Fauvism,High_Renaissance,Mannerism_Late_Renaissance,Naive_Art_Primitivism,Northern_Renaissance,Pop_Art,Post_Impressionism,Romanticism,Symbolism"

New-Item -ItemType Directory -Force -Path "$REPO\logs" | Out-Null
New-Item -ItemType Directory -Force -Path $OUT_DIR | Out-Null

Set-Location $SRC_DIR

"=== wikiarts-15 WD-VF eval started at $(Get-Date) ===" | Tee-Object -FilePath $LOG -Append
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
