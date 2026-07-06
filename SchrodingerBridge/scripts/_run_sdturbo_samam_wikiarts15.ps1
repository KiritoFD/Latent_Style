# Run SD-Turbo and SaMam generation + evaluation on WikiArt-15
# Serial execution (VRAM constraint: 12GB RTX 3060)

$REPO = "I:\Github\Latent_Style\SchrodingerBridge"
$PYTHON = "C:\Program Files\Python312\python.exe"
$SRC_DIR = "$REPO\src"
$SCRIPTS_DIR = "$REPO\scripts"
$TEST_DIR = "I:\datasets\wikiarts15_512_test"
$OUT_ROOT = "$REPO\exp\baseline_wikiarts15"
$LOG = "$REPO\logs\wikiarts15_sdturbo_samam.log"

$USER_SITE = "C:\Users\Administrator\AppData\Roaming\Python\Python312\site-packages"
$env:PYTHONPATH = "$SRC_DIR;$USER_SITE"
$env:PYTHONUSERBASE = "C:\Users\Administrator\AppData\Roaming\Python"
$env:CUDA_VISIBLE_DEVICES = "0"
$env:HF_HOME = "$REPO\exp\eval_cache\hf"
$env:TRANSFORMERS_OFFLINE = "0"

$STYLES = "Abstract_Expressionism,Art_Nouveau_Modern,Baroque,Color_Field_Painting,Cubism,Expressionism,Fauvism,High_Renaissance,Mannerism_Late_Renaissance,Naive_Art_Primitivism,Northern_Renaissance,Pop_Art,Post_Impressionism,Romanticism,Symbolism"

New-Item -ItemType Directory -Force -Path "$REPO\logs" | Out-Null
New-Item -ItemType Directory -Force -Path $OUT_ROOT | Out-Null

"=== WikiArt-15 SD-Turbo + SaMam pipeline started at $(Get-Date) ===" | Tee-Object -FilePath $LOG -Append

# ── Step 1: SD-Turbo generation ──
"--- Step 1: SD-Turbo generation ---" | Tee-Object -FilePath $LOG -Append
$sdturboDone = "$OUT_ROOT\sdturbo\_DONE"
if (Test-Path $sdturboDone) {
    "  [sdturbo] _DONE marker found, skipping generation" | Tee-Object -FilePath $LOG -Append
} else {
    "  [sdturbo] Generating at $(Get-Date)..." | Tee-Object -FilePath $LOG -Append
    $genArgs = @("-u", "$SCRIPTS_DIR\gen_sdturbo_wikiarts15.py")
    $proc = Start-Process -FilePath $PYTHON -ArgumentList $genArgs -NoNewWindow -PassThru -WorkingDirectory $REPO -RedirectStandardOutput "$LOG.sdturbo.gen.out" -RedirectStandardError "$LOG.sdturbo.gen.err"
    $proc.WaitForExit()
    if ($proc.ExitCode -eq 0) {
        "  [sdturbo] GEN SUCCESS at $(Get-Date)" | Tee-Object -FilePath $LOG -Append
    } else {
        "  [sdturbo] GEN FAIL exit=$($proc.ExitCode) at $(Get-Date)" | Tee-Object -FilePath $LOG -Append
        if (Test-Path "$LOG.sdturbo.gen.err") {
            Get-Content "$LOG.sdturbo.gen.err" -Tail 30 | ForEach-Object { "    $_" } | Tee-Object -FilePath $LOG -Append
        }
    }
}

# ── Step 2: SD-Turbo evaluation ──
"--- Step 2: SD-Turbo evaluation ---" | Tee-Object -FilePath $LOG -Append
if (Test-Path $sdturboDone) {
    $evalDir = "$OUT_ROOT\sdturbo"
    $evalArgs = @("-u", "utils\run_evaluation.py", $evalDir, "--reuse_generated", "--style_subdirs", $STYLES, "--test_dir", $TEST_DIR, "--eval_only_lpips_clip_style", "--max_src_samples", "30", "--batch_size", "2", "--ref_feature_batch_size", "2")
    $proc = Start-Process -FilePath $PYTHON -ArgumentList $evalArgs -NoNewWindow -PassThru -WorkingDirectory $SRC_DIR -RedirectStandardOutput "$LOG.sdturbo.eval.out" -RedirectStandardError "$LOG.sdturbo.eval.err"
    $proc.WaitForExit()
    if ($proc.ExitCode -eq 0) {
        "  [sdturbo] EVAL SUCCESS at $(Get-Date)" | Tee-Object -FilePath $LOG -Append
    } else {
        "  [sdturbo] EVAL FAIL exit=$($proc.ExitCode) at $(Get-Date)" | Tee-Object -FilePath $LOG -Append
        if (Test-Path "$LOG.sdturbo.eval.err") {
            Get-Content "$LOG.sdturbo.eval.err" -Tail 30 | ForEach-Object { "    $_" } | Tee-Object -FilePath $LOG -Append
        }
    }
}

# ── Step 3: SaMam generation ──
"--- Step 3: SaMam generation ---" | Tee-Object -FilePath $LOG -Append
$samamDone = "$OUT_ROOT\samam\_DONE"
if (Test-Path $samamDone) {
    "  [samam] _DONE marker found, skipping generation" | Tee-Object -FilePath $LOG -Append
} else {
    "  [samam] Generating at $(Get-Date)..." | Tee-Object -FilePath $LOG -Append
    $genArgs = @("-u", "$SCRIPTS_DIR\gen_samam_wikiarts15.py")
    $proc = Start-Process -FilePath $PYTHON -ArgumentList $genArgs -NoNewWindow -PassThru -WorkingDirectory $REPO -RedirectStandardOutput "$LOG.samam.gen.out" -RedirectStandardError "$LOG.samam.gen.err"
    $proc.WaitForExit()
    if ($proc.ExitCode -eq 0) {
        "  [samam] GEN SUCCESS at $(Get-Date)" | Tee-Object -FilePath $LOG -Append
    } else {
        "  [samam] GEN FAIL exit=$($proc.ExitCode) at $(Get-Date)" | Tee-Object -FilePath $LOG -Append
        if (Test-Path "$LOG.samam.gen.err") {
            Get-Content "$LOG.samam.gen.err" -Tail 30 | ForEach-Object { "    $_" } | Tee-Object -FilePath $LOG -Append
        }
    }
}

# ── Step 4: SaMam evaluation ──
"--- Step 4: SaMam evaluation ---" | Tee-Object -FilePath $LOG -Append
if (Test-Path $samamDone) {
    $evalDir = "$OUT_ROOT\samam"
    $evalArgs = @("-u", "utils\run_evaluation.py", $evalDir, "--reuse_generated", "--style_subdirs", $STYLES, "--test_dir", $TEST_DIR, "--eval_only_lpips_clip_style", "--max_src_samples", "30", "--batch_size", "2", "--ref_feature_batch_size", "2")
    $proc = Start-Process -FilePath $PYTHON -ArgumentList $evalArgs -NoNewWindow -PassThru -WorkingDirectory $SRC_DIR -RedirectStandardOutput "$LOG.samam.eval.out" -RedirectStandardError "$LOG.samam.eval.err"
    $proc.WaitForExit()
    if ($proc.ExitCode -eq 0) {
        "  [samam] EVAL SUCCESS at $(Get-Date)" | Tee-Object -FilePath $LOG -Append
    } else {
        "  [samam] EVAL FAIL exit=$($proc.ExitCode) at $(Get-Date)" | Tee-Object -FilePath $LOG -Append
        if (Test-Path "$LOG.samam.eval.err") {
            Get-Content "$LOG.samam.eval.err" -Tail 30 | ForEach-Object { "    $_" } | Tee-Object -FilePath $LOG -Append
        }
    }
}

"=== Pipeline finished at $(Get-Date) ===" | Tee-Object -FilePath $LOG -Append
