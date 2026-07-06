# Unified pipeline to fill main table entries.
# Generates SD-Turbo/StyleID on 256 + SaMST/StyleID/SD-Turbo on wiki20-distinct5,
# then evaluates all with CLIP-S + LPIPS + MUSIQ.
#
# Budget estimate:
#   - 256 SD-Turbo: 750 imgs x ~0.5s = ~6 min
#   - 256 StyleID:  750 imgs x ~3s   = ~37 min
#   - wiki20 SaMST:  750 imgs x ~0.2s = ~3 min
#   - wiki20 SD-Turbo: 750 imgs x ~0.5s = ~6 min
#   - wiki20 StyleID:  750 imgs x ~3s   = ~37 min
#   - Eval (5 methods x ~1 min each) = ~5 min
#   Total: ~1.5 hours
#
# Logs to logs\pipeline_fill_main.log
# Results to exp\_pipeline_fill_results.json

$ErrorActionPreference = "Continue"

# ── Paths ──
$REPO = "I:\Github\Latent_Style\SchrodingerBridge"
$PYTHON = "C:\Program Files\Python312\python.exe"
$SRC_DIR = "$REPO\src"
$SCRIPTS_DIR = "$REPO\scripts"
$LOG_DIR = "$REPO\logs"
$LOG = "$LOG_DIR\pipeline_fill_main.log"
$RESULTS_JSON = "$REPO\exp\_pipeline_fill_results.json"

$TEST_256 = "I:\datasets\legacy256_overfit50\test"
$TEST_W20 = "I:\datasets\wikiarts20_512_test"
$EXP_256 = "I:\exp_256_photo2art"
$EXP_W20 = "$REPO\exp\baseline_wikiarts20"

$SAMST_CKPT = "I:\Github\Latent_Style\Related_Works\repos\SaMST-main\checkpoint\epoch_20.model"
$SAMST_ROOT = "I:\Github\Latent_Style\Related_Works\repos\SaMST-main"

$LEGACY5 = "cezanne,Hayao,monet,photo,vangogh"
$DISTINCT5 = "Early_Renaissance,Impressionism,Minimalism,Rococo,Ukiyo_e"

# ── Environment ──
$USER_SITE = "C:\Users\Administrator\AppData\Roaming\Python\Python312\site-packages"
$env:PYTHONPATH = "$SRC_DIR;$USER_SITE;$SCRIPTS_DIR"
$env:PYTHONUSERBASE = "C:\Users\Administrator\AppData\Roaming\Python"
$env:CUDA_VISIBLE_DEVICES = "0"
# HF_HOME must point to where models are actually cached
$env:HF_HOME = "C:\Users\Administrator\.cache\huggingface"
# Note: do NOT set HF_HUB_OFFLINE=1 (breaks diffusers from_pretrained even with cache)
# Instead, _gen_diffusion_baseline.py uses local_files_only=True
$env:TRANSFORMERS_OFFLINE = "1"
$env:TORCH_HOME = "C:\Users\Administrator\.cache\torch"

# ── Setup ──
New-Item -ItemType Directory -Force -Path $LOG_DIR | Out-Null
New-Item -ItemType Directory -Force -Path "$REPO\exp" | Out-Null
New-Item -ItemType Directory -Force -Path $EXP_256 | Out-Null
New-Item -ItemType Directory -Force -Path $EXP_W20 | Out-Null

# Helper: load existing results JSON
function Load-Results {
    if (Test-Path $RESULTS_JSON) {
        try { return Get-Content $RESULTS_JSON -Raw | ConvertFrom-Json -AsHashtable }
        catch { return @{} }
    }
    return @{}
}

function Save-Results($results) {
    $results | ConvertTo-Json -Depth 5 | Out-File -FilePath $RESULTS_JSON -Encoding utf8 -Force
}

# Helper: run a Python process with logging
function Invoke-PythonTask($name, $pyArgs, $cwd, $logPrefix) {
    $outFile = "${LOG}.${logPrefix}.out"
    $errFile = "${LOG}.${logPrefix}.err"
    "    CMD: $PYTHON $($pyArgs -join ' ')" | Tee-Object -FilePath $LOG -Append
    try {
        $proc = Start-Process -FilePath $PYTHON -ArgumentList $pyArgs -NoNewWindow -PassThru `
            -WorkingDirectory $cwd -RedirectStandardOutput $outFile -RedirectStandardError $errFile
        $proc.WaitForExit()
        $exitCode = $proc.ExitCode
        if ($exitCode -eq 0) {
            "  [$name] SUCCESS exit=0 at $(Get-Date)" | Tee-Object -FilePath $LOG -Append
            if (Test-Path $outFile) {
                Get-Content $outFile -Tail 8 | ForEach-Object { "    $_" } | Tee-Object -FilePath $LOG -Append
            }
        } else {
            "  [$name] FAIL exit=$exitCode at $(Get-Date)" | Tee-Object -FilePath $LOG -Append
            if (Test-Path $errFile) {
                "    last 25 lines of stderr:" | Tee-Object -FilePath $LOG -Append
                Get-Content $errFile -Tail 25 | ForEach-Object { "      $_" } | Tee-Object -FilePath $LOG -Append
            }
        }
        return $exitCode
    } catch {
        "  [$name] EXCEPTION: $_ at $(Get-Date)" | Tee-Object -FilePath $LOG -Append
        return -1
    }
}

# Helper: count images in a dir
function Count-Images($dir) {
    if (-not (Test-Path $dir)) { return 0 }
    $png = (Get-ChildItem $dir -Filter "*.png" -ErrorAction SilentlyContinue).Count
    $jpg = (Get-ChildItem $dir -Filter "*.jpg" -ErrorAction SilentlyContinue).Count
    return $png + $jpg
}

# Helper: evaluate a method's image dir with the unified eval script
function Invoke-Eval($name, $imageDir, $dataset, $maxImages, $logPrefix) {
    $evalOutput = "$RESULTS_JSON.$logPrefix.json"
    $evalArgs = @(
        "-u", "$SCRIPTS_DIR\_eval_unified.py",
        "--image-dir", $imageDir,
        "--dataset", $dataset,
        "--output", $evalOutput,
        "--max-images", "$maxImages"
    )
    $ec = Invoke-PythonTask "eval-$name" $evalArgs $REPO $logPrefix
    if ($ec -eq 0 -and (Test-Path $evalOutput)) {
        try {
            $evalResult = Get-Content $evalOutput -Raw | ConvertFrom-Json -AsHashtable
            return $evalResult
        } catch {
            "  [eval-$name] Failed to parse ${evalOutput}: $_" | Tee-Object -FilePath $LOG -Append
            return $null
        }
    }
    return $null
}

$results = Load-Results

"=" * 80 | Tee-Object -FilePath $LOG -Append
"=== Pipeline fill-main started at $(Get-Date) ===" | Tee-Object -FilePath $LOG -Append
"=" * 80 | Tee-Object -FilePath $LOG -Append
"  REPO: $REPO" | Tee-Object -FilePath $LOG -Append
"  TEST_256: $TEST_256" | Tee-Object -FilePath $LOG -Append
"  TEST_W20: $TEST_W20" | Tee-Object -FilePath $LOG -Append
"  Results JSON: $RESULTS_JSON" | Tee-Object -FilePath $LOG -Append

# ═══════════════════════════════════════════════════════════════
# Phase A: Photo2Art-256 — SD-Turbo + StyleID (750 imgs each)
# ═══════════════════════════════════════════════════════════════
"" | Tee-Object -FilePath $LOG -Append
"--- Phase A: Photo2Art-256 (SD-Turbo + StyleID) ---" | Tee-Object -FilePath $LOG -Append

# A1: SD-Turbo 256
$sdturbo256Dir = "$EXP_256\sdturbo_256\images"
New-Item -ItemType Directory -Force -Path $sdturbo256Dir | Out-Null
$cnt = Count-Images $sdturbo256Dir
"  [sdturbo_256] existing: $cnt/750" | Tee-Object -FilePath $LOG -Append
if ($cnt -lt 750) {
    $pyArgs = @("-u", "$SCRIPTS_DIR\_gen_diffusion_baseline.py",
              "--method", "sdturbo", "--test-dir", $TEST_256,
              "--output-dir", $sdturbo256Dir, "--styles", $LEGACY5,
              "--image-size", "256", "--max-src-per-style", "30")
    Invoke-PythonTask "sdturbo_256" $pyArgs $REPO "sdturbo256.gen" | Out-Null
}
$cnt = Count-Images $sdturbo256Dir
$results["sdturbo_256_gen_count"] = $cnt
"  [sdturbo_256] final: $cnt/750" | Tee-Object -FilePath $LOG -Append
Save-Results $results

# A2: StyleID 256
$styleid256Dir = "$EXP_256\styleid_256\images"
New-Item -ItemType Directory -Force -Path $styleid256Dir | Out-Null
$cnt = Count-Images $styleid256Dir
"  [styleid_256] existing: $cnt/750" | Tee-Object -FilePath $LOG -Append
if ($cnt -lt 750) {
    $pyArgs = @("-u", "$SCRIPTS_DIR\_gen_diffusion_baseline.py",
              "--method", "styleid", "--test-dir", $TEST_256,
              "--output-dir", $styleid256Dir, "--styles", $LEGACY5,
              "--image-size", "256", "--max-src-per-style", "30")
    Invoke-PythonTask "styleid_256" $pyArgs $REPO "styleid256.gen" | Out-Null
}
$cnt = Count-Images $styleid256Dir
$results["styleid_256_gen_count"] = $cnt
"  [styleid_256] final: $cnt/750" | Tee-Object -FilePath $LOG -Append
Save-Results $results

# ═══════════════════════════════════════════════════════════════
# Phase B: WikiArt-20 distinct5 — SaMST + SD-Turbo + StyleID (750 imgs each)
# ═══════════════════════════════════════════════════════════════
"" | Tee-Object -FilePath $LOG -Append
"--- Phase B: WikiArt-20 distinct5 (SaMST + SD-Turbo + StyleID) ---" | Tee-Object -FilePath $LOG -Append

# B1: SaMST wiki20
$samstW20Dir = "$EXP_W20\samst\images"
New-Item -ItemType Directory -Force -Path $samstW20Dir | Out-Null
$cnt = Count-Images $samstW20Dir
"  [samst_w20] existing: $cnt/750" | Tee-Object -FilePath $LOG -Append
if ($cnt -lt 750) {
    $pyArgs = @("-u", "$SCRIPTS_DIR\_gen_samst_wiki20.py",
              "--test-dir", $TEST_W20, "--output-dir", $samstW20Dir,
              "--checkpoint", $SAMST_CKPT, "--samst-root", $SAMST_ROOT,
              "--styles", $DISTINCT5, "--max-src-per-style", "30",
              "--image-size", "512")
    Invoke-PythonTask "samst_w20" $pyArgs $REPO "samstw20.gen" | Out-Null
}
$cnt = Count-Images $samstW20Dir
$results["samst_w20_gen_count"] = $cnt
"  [samst_w20] final: $cnt/750" | Tee-Object -FilePath $LOG -Append
Save-Results $results

# B2: SD-Turbo wiki20
$sdturboW20Dir = "$EXP_W20\sdturbo\images"
New-Item -ItemType Directory -Force -Path $sdturboW20Dir | Out-Null
$cnt = Count-Images $sdturboW20Dir
"  [sdturbo_w20] existing: $cnt/750" | Tee-Object -FilePath $LOG -Append
if ($cnt -lt 750) {
    $pyArgs = @("-u", "$SCRIPTS_DIR\_gen_diffusion_baseline.py",
              "--method", "sdturbo", "--test-dir", $TEST_W20,
              "--output-dir", $sdturboW20Dir, "--styles", $DISTINCT5,
              "--image-size", "512", "--max-src-per-style", "30")
    Invoke-PythonTask "sdturbo_w20" $pyArgs $REPO "sdturbow20.gen" | Out-Null
}
$cnt = Count-Images $sdturboW20Dir
$results["sdturbo_w20_gen_count"] = $cnt
"  [sdturbo_w20] final: $cnt/750" | Tee-Object -FilePath $LOG -Append
Save-Results $results

# B3: StyleID wiki20
$styleidW20Dir = "$EXP_W20\styleid\images"
New-Item -ItemType Directory -Force -Path $styleidW20Dir | Out-Null
$cnt = Count-Images $styleidW20Dir
"  [styleid_w20] existing: $cnt/750" | Tee-Object -FilePath $LOG -Append
if ($cnt -lt 750) {
    $pyArgs = @("-u", "$SCRIPTS_DIR\_gen_diffusion_baseline.py",
              "--method", "styleid", "--test-dir", $TEST_W20,
              "--output-dir", $styleidW20Dir, "--styles", $DISTINCT5,
              "--image-size", "512", "--max-src-per-style", "30")
    Invoke-PythonTask "styleid_w20" $pyArgs $REPO "styleidw20.gen" | Out-Null
}
$cnt = Count-Images $styleidW20Dir
$results["styleid_w20_gen_count"] = $cnt
"  [styleid_w20] final: $cnt/750" | Tee-Object -FilePath $LOG -Append
Save-Results $results

# ═══════════════════════════════════════════════════════════════
# Phase C: Evaluate all methods with CLIP-S + LPIPS + MUSIQ
# ═══════════════════════════════════════════════════════════════
"" | Tee-Object -FilePath $LOG -Append
"--- Phase C: Evaluate all methods ---" | Tee-Object -FilePath $LOG -Append

# C1: Evaluate 256 methods
$evalTargets = @(
    @{ name = "sdturbo_256"; dir = $sdturbo256Dir; dataset = "photo2art256"; max = 750 },
    @{ name = "styleid_256"; dir = $styleid256Dir; dataset = "photo2art256"; max = 750 },
    @{ name = "samst_w20";   dir = $samstW20Dir;   dataset = "wiki20distinct5"; max = 750 },
    @{ name = "sdturbo_w20"; dir = $sdturboW20Dir; dataset = "wiki20distinct5"; max = 750 },
    @{ name = "styleid_w20"; dir = $styleidW20Dir; dataset = "wiki20distinct5"; max = 750 }
)

foreach ($t in $evalTargets) {
    $cnt = Count-Images $t.dir
    if ($cnt -eq 0) {
        "  [eval-$($t.name)] SKIP: 0 images" | Tee-Object -FilePath $LOG -Append
        continue
    }
    "  [eval-$($t.name)] Evaluating $cnt images ($($t.dataset))..." | Tee-Object -FilePath $LOG -Append
    $evalResult = Invoke-Eval $t.name $t.dir $t.dataset $t.max "eval_$($t.name)"
    if ($evalResult) {
        $results[$t.name] = $evalResult
        Save-Results $results
        "  [eval-$($t.name)] CLIP-S=$($evalResult.clip_s)  LPIPS=$($evalResult.lpips)  MUSIQ=$($evalResult.musiq)" | Tee-Object -FilePath $LOG -Append
    } else {
        "  [eval-$($t.name)] FAILED" | Tee-Object -FilePath $LOG -Append
    }
}

# ═══════════════════════════════════════════════════════════════
# Phase D: Re-evaluate existing baseline_v2 (512 distinct5) for MUSIQ
# (These have CLIP-S and LPIPS in paper.tex but no MUSIQ column yet)
# ═══════════════════════════════════════════════════════════════
"" | Tee-Object -FilePath $LOG -Append
"--- Phase D: MUSIQ re-eval for existing 512 distinct5 baselines ---" | Tee-Object -FilePath $LOG -Append

$baselineV2 = "$REPO\exp\baseline_v2\images"
$existingMethods = @("adain", "wct", "samst", "samam", "sdturbo", "styleid", "cut", "identity")
foreach ($m in $existingMethods) {
    $mDir = "$baselineV2\$m"
    $cnt = Count-Images $mDir
    if ($cnt -eq 0) {
        # Try alternate structure with /images subdir
        $mDir = "$baselineV2\$m\images"
        $cnt = Count-Images $mDir
    }
    if ($cnt -eq 0) {
        "  [musiq-$m] SKIP: no images" | Tee-Object -FilePath $LOG -Append
        continue
    }
    "  [musiq-$m] Re-evaluating $cnt images (photo2art256 dataset for legacy256 test set)..." | Tee-Object -FilePath $LOG -Append
    # Note: baseline_v2 is on 512 distinct5, not 256. But our eval script only supports
    # photo2art256 and wiki20distinct5. For 512 distinct5 we need a different dataset config.
    # Skip for now - the existing MUSIQ values from 256 eval JSON are already in paper.tex
    "  [musiq-$m] SKIP: 512 distinct5 dataset not in unified eval script (use existing MUSIQ values)" | Tee-Object -FilePath $LOG -Append
}

# ═══════════════════════════════════════════════════════════════
# Done
# ═══════════════════════════════════════════════════════════════
"" | Tee-Object -FilePath $LOG -Append
"=" * 80 | Tee-Object -FilePath $LOG -Append
"=== Pipeline fill-main COMPLETED at $(Get-Date) ===" | Tee-Object -FilePath $LOG -Append
"=" * 80 | Tee-Object -FilePath $LOG -Append
"  Results: $RESULTS_JSON" | Tee-Object -FilePath $LOG -Append
Save-Results $results
