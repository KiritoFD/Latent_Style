# Phase 1: WikiArt-20 baselines — StyleID, CUT, SaMST
# Generates 12000 images per method (20 styles x 20 styles x 30 src) on wikiarts20_512_test,
# then evaluates each with CLIP-S + LPIPS via run_evaluation.py --reuse_generated.
#
# - StyleID: SD1.5 img2img (training-free). Output to exp\baseline_wikiarts20\styleid\
# - CUT: needs per-style training (20 styles too expensive). Skipped with "--" note if no checkpoints.
# - SaMST: uses existing 5-style checkpoint on distinct5 subset (5x5x30=750 images).
#
# Skip-resumable: existing output PNGs are skipped.
# Logs to logs\phase1_wiki20_scm.log
# Aggregated results written to exp\_baseline_fill_results.json

$ErrorActionPreference = "Continue"

# ── Paths ──
$REPO = "I:\Github\Latent_Style\SchrodingerBridge"
$PYTHON = "C:\Program Files\Python312\python.exe"
$SRC_DIR = "$REPO\src"
$SCRIPTS_DIR = "$REPO\scripts"
$TEST_DIR = "I:\datasets\wikiarts20_512_test"
$OUT_ROOT = "$REPO\exp\baseline_wikiarts20"
$LOG_DIR = "$REPO\logs"
$LOG = "$LOG_DIR\phase1_wiki20_scm.log"
$RESULTS_JSON = "$REPO\exp\_baseline_fill_results.json"

# SaMST paths
$SAMST_REPO = "I:\Github\Latent_Style\Related_Works\repos\SaMST-main"
$SAMST_CKPT = "$SAMST_REPO\checkpoint\epoch_20.model"

# CUT paths
$CUT_REPO = "I:\Github\Latent_Style\Related_Works\repos\external\CUT"

# ── Environment ──
$USER_SITE = "C:\Users\Administrator\AppData\Roaming\Python\Python312\site-packages"
$env:PYTHONPATH = "$SRC_DIR;$USER_SITE"
$env:PYTHONUSERBASE = "C:\Users\Administrator\AppData\Roaming\Python"
$env:CUDA_VISIBLE_DEVICES = "0"
$env:HF_HOME = "$REPO\exp\eval_cache\hf"
$env:TRANSFORMERS_OFFLINE = "0"

# ── WikiArt-20 styles ──
$WIKI20_STYLES = "Abstract_Expressionism,Art_Nouveau_Modern,Baroque,Color_Field_Painting,Cubism,Early_Renaissance,Expressionism,Fauvism,High_Renaissance,Impressionism,Mannerism_Late_Renaissance,Minimalism,Naive_Art_Primitivism,Northern_Renaissance,Pop_Art,Post_Impressionism,Rococo,Romanticism,Symbolism,Ukiyo_e"

# Distinct5 subset (for SaMST — 5-style checkpoint)
$DISTINCT5 = "Early_Renaissance,Impressionism,Minimalism,Rococo,Ukiyo_e"

# ── Setup ──
New-Item -ItemType Directory -Force -Path $LOG_DIR | Out-Null
New-Item -ItemType Directory -Force -Path $OUT_ROOT | Out-Null
New-Item -ItemType Directory -Force -Path "$REPO\exp" | Out-Null

# Helper: load existing results JSON (or create empty)
function Load-Results {
    if (Test-Path $RESULTS_JSON) {
        try {
            return Get-Content $RESULTS_JSON -Raw | ConvertFrom-Json -AsHashtable
        } catch {
            return @{}
        }
    }
    return @{}
}

function Save-Results($results) {
    $results | ConvertTo-Json -Depth 5 | Out-File -FilePath $RESULTS_JSON -Encoding utf8 -Force
}

# Helper: run a Python process and log output
function Invoke-PythonTask($name, $pyArgs, $cwd, $logPrefix) {
    $outFile = "$LOG.$logPrefix.out"
    $errFile = "$LOG.$logPrefix.err"
    "    CMD: $PYTHON $($pyArgs -join ' ')" | Tee-Object -FilePath $LOG -Append
    try {
        $proc = Start-Process -FilePath $PYTHON -ArgumentList $pyArgs -NoNewWindow -PassThru `
            -WorkingDirectory $cwd -RedirectStandardOutput $outFile -RedirectStandardError $errFile
        $proc.WaitForExit()
        $exitCode = $proc.ExitCode
        if ($exitCode -eq 0) {
            "  [$name] SUCCESS exit=0 at $(Get-Date)" | Tee-Object -FilePath $LOG -Append
            if (Test-Path $outFile) {
                Get-Content $outFile -Tail 5 | ForEach-Object { "    $_" } | Tee-Object -FilePath $LOG -Append
            }
        } else {
            "  [$name] FAIL exit=$exitCode at $(Get-Date)" | Tee-Object -FilePath $LOG -Append
            if (Test-Path $errFile) {
                "    last 20 lines of stderr:" | Tee-Object -FilePath $LOG -Append
                Get-Content $errFile -Tail 20 | ForEach-Object { "      $_" } | Tee-Object -FilePath $LOG -Append
            }
        }
        return $exitCode
    } catch {
        "  [$name] EXCEPTION: $_ at $(Get-Date)" | Tee-Object -FilePath $LOG -Append
        return -1
    }
}

$results = Load-Results
if (-not $results.ContainsKey("phase1_wiki20")) {
    $results["phase1_wiki20"] = @{}
}

"=== Phase 1: WikiArt-20 StyleID/CUT/SaMST started at $(Get-Date) ===" | Tee-Object -FilePath $LOG -Append
"  test_dir: $TEST_DIR" | Tee-Object -FilePath $LOG -Append
"  out_root: $OUT_ROOT" | Tee-Object -FilePath $LOG -Append
"  wiki20_styles: $WIKI20_STYLES" | Tee-Object -FilePath $LOG -Append
"  distinct5: $DISTINCT5" | Tee-Object -FilePath $LOG -Append

# ═══════════════════════════════════════════════════════════════
# Step 1: StyleID — SD1.5 img2img on wiki20 (12000 images)
# ═══════════════════════════════════════════════════════════════
"--- Step 1: StyleID (SD1.5 img2img, 12000 images) ---" | Tee-Object -FilePath $LOG -Append

try {
    $styleidDir = "$OUT_ROOT\styleid"
    $styleidImages = "$styleidDir\images"
    New-Item -ItemType Directory -Force -Path $styleidImages | Out-Null

    $styleidArgs = @(
        "-u",
        "$SCRIPTS_DIR\_gen_diffusion_baseline.py",
        "--method", "styleid",
        "--test-dir", $TEST_DIR,
        "--output-dir", $styleidImages,
        "--styles", $WIKI20_STYLES,
        "--image-size", "512",
        "--max-src-per-style", "30"
    )

    $ec = Invoke-PythonTask "styleid-gen" $styleidArgs $REPO "styleid.gen"
    $imgCount = 0
    if (Test-Path $styleidImages) {
        $imgCount = (Get-ChildItem $styleidImages -Filter "*.png" -ErrorAction SilentlyContinue).Count
    }
    "  [styleid] image count: $imgCount (expected 12000)" | Tee-Object -FilePath $LOG -Append
    $results["phase1_wiki20"]["styleid_gen_count"] = $imgCount
    Save-Results $results
} catch {
    "  [styleid] GEN EXCEPTION: $_ at $(Get-Date)" | Tee-Object -FilePath $LOG -Append
    $results["phase1_wiki20"]["styleid_error"] = "$_"
    Save-Results $results
}

# ═══════════════════════════════════════════════════════════════
# Step 2: CUT — needs per-style training; skip with note if no checkpoints
# ═══════════════════════════════════════════════════════════════
"--- Step 2: CUT (per-style training required) ---" | Tee-Object -FilePath $LOG -Append

try {
    $cutCkptDir = "$CUT_REPO\checkpoints"
    $cutCkpts = @()
    if (Test-Path $cutCkptDir) {
        $cutCkpts = Get-ChildItem $cutCkptDir -Recurse -Filter "*.pth" -ErrorAction SilentlyContinue
    }

    if ($cutCkpts.Count -eq 0) {
        "  [cut] No pretrained checkpoints found at $cutCkptDir" | Tee-Object -FilePath $LOG -Append
        "  [cut] CUT requires per-style training (too expensive for 20 styles). Marking as '--'." | Tee-Object -FilePath $LOG -Append
        $results["phase1_wiki20"]["cut"] = @{ status = "skipped"; reason = "no pretrained checkpoints; per-style training too expensive for 20 styles"; musiq = $null; clip_s = $null; lpips = $null }
        Save-Results $results
    } else {
        "  [cut] Found $($cutCkpts.Count) checkpoint(s). Attempting CUT inference..." | Tee-Object -FilePath $LOG -Append
        # If checkpoints exist, would run CUT test.py here. For now, mark as partial.
        $results["phase1_wiki20"]["cut"] = @{ status = "partial"; reason = "checkpoints exist but CUT 20-style inference not implemented in this script"; n_ckpts = $cutCkpts.Count }
        Save-Results $results
    }
} catch {
    "  [cut] EXCEPTION: $_ at $(Get-Date)" | Tee-Object -FilePath $LOG -Append
    $results["phase1_wiki20"]["cut"] = @{ status = "error"; error = "$_" }
    Save-Results $results
}

# ═══════════════════════════════════════════════════════════════
# Step 3: SaMST — existing 5-style checkpoint on distinct5 subset (750 images)
# ═══════════════════════════════════════════════════════════════
"--- Step 3: SaMST (5-style checkpoint on distinct5 subset) ---" | Tee-Object -FilePath $LOG -Append

try {
    $samstDir = "$OUT_ROOT\samst"
    $samstImages = "$samstDir\images"
    New-Item -ItemType Directory -Force -Path $samstImages | Out-Null

    if (-not (Test-Path $SAMST_CKPT)) {
        "  [samst] Checkpoint not found: $SAMST_CKPT" | Tee-Object -FilePath $LOG -Append
        "  [samst] Marking as '--'." | Tee-Object -FilePath $LOG -Append
        $results["phase1_wiki20"]["samst"] = @{ status = "skipped"; reason = "checkpoint not found"; musiq = $null; clip_s = $null; lpips = $null }
        Save-Results $results
    } else {
        $samstArgs = @(
            "-u",
            "$SCRIPTS_DIR\_gen_samst_wiki20.py",
            "--test-dir", $TEST_DIR,
            "--output-dir", $samstImages,
            "--checkpoint", $SAMST_CKPT,
            "--samst-root", $SAMST_REPO,
            "--styles", $DISTINCT5,
            "--style-num", "5",
            "--max-src-per-style", "30",
            "--image-size", "512"
        )

        $ec = Invoke-PythonTask "samst-gen" $samstArgs $SAMST_REPO "samst.gen"
        $imgCount = 0
        if (Test-Path $samstImages) {
            $imgCount = (Get-ChildItem $samstImages -Filter "*.png" -ErrorAction SilentlyContinue).Count
        }
        "  [samst] image count: $imgCount (expected 750 for 5x5x30 distinct5)" | Tee-Object -FilePath $LOG -Append
        $results["phase1_wiki20"]["samst_gen_count"] = $imgCount
        $results["phase1_wiki20"]["samst_subset"] = "distinct5"
        Save-Results $results
    }
} catch {
    "  [samst] GEN EXCEPTION: $_ at $(Get-Date)" | Tee-Object -FilePath $LOG -Append
    $results["phase1_wiki20"]["samst"] = @{ status = "error"; error = "$_" }
    Save-Results $results
}

# ═══════════════════════════════════════════════════════════════
# Step 4: Evaluate each method (CLIP-S + LPIPS via run_evaluation.py)
# ═══════════════════════════════════════════════════════════════
"--- Step 4: Evaluate methods (CLIP-S + LPIPS) ---" | Tee-Object -FilePath $LOG -Append

$methodsToEval = @(
    @{ name = "styleid"; dir = "$OUT_ROOT\styleid"; styles = $WIKI20_STYLES; expected = 12000 },
    @{ name = "samst"; dir = "$OUT_ROOT\samst"; styles = $DISTINCT5; expected = 750 }
)

foreach ($m in $methodsToEval) {
    $mName = $m.name
    $mDir = $m.dir
    $mStyles = $m.styles
    $mExpected = $m.expected

    "  [$mName] Evaluating at $(Get-Date)..." | Tee-Object -FilePath $LOG -Append

    # Check _DONE marker
    $doneMarker = "$mDir\_DONE"
    if (-not (Test-Path $doneMarker)) {
        "  [$mName] EVAL SKIP: _DONE marker not found (generation incomplete)" | Tee-Object -FilePath $LOG -Append
        $results["phase1_wiki20"][$mName] = @{ status = "incomplete"; musiq = $null; clip_s = $null; lpips = $null }
        Save-Results $results
        continue
    }

    # Count images
    $imgCount = (Get-ChildItem (Join-Path $mDir "images") -Filter "*.png" -ErrorAction SilentlyContinue).Count
    "  [$mName] image count: $imgCount (expected $mExpected)" | Tee-Object -FilePath $LOG -Append

    if ($imgCount -lt 1) {
        "  [$mName] EVAL SKIP: no images" | Tee-Object -FilePath $LOG -Append
        $results["phase1_wiki20"][$mName] = @{ status = "no images"; musiq = $null; clip_s = $null; lpips = $null }
        Save-Results $results
        continue
    }

    $evalArgs = @(
        "-u",
        "utils\run_evaluation.py",
        $mDir,
        "--reuse_generated",
        "--style_subdirs", $mStyles,
        "--test_dir", $TEST_DIR,
        "--eval_only_lpips_clip_style",
        "--max_src_samples", "30",
        "--batch_size", "2",
        "--ref_feature_batch_size", "2"
    )

    $ec = Invoke-PythonTask "$mName-eval" $evalArgs $SRC_DIR "$mName.eval"

    # Read metrics.csv if produced
    $metricsCsv = "$mDir\metrics.csv"
    if (Test-Path $metricsCsv) {
        try {
            $csvContent = Import-Csv $metricsCsv
            $clipSSum = 0.0; $lpipsSum = 0.0; $n = 0
            foreach ($row in $csvContent) {
                try {
                    $lpipsVal = [double]$row.content_lpips
                    $clipSVal = [double]$row.clip_style
                    $clipSSum += $clipSVal
                    $lpipsSum += $lpipsVal
                    $n++
                } catch { }
            }
            if ($n -gt 0) {
                $clipSAvg = $clipSSum / $n
                $lpipsAvg = $lpipsSum / $n
                "  [$mName] CLIP-S=$clipSAvg  LPIPS=$lpipsAvg  (n=$n)" | Tee-Object -FilePath $LOG -Append
                $results["phase1_wiki20"][$mName] = @{
                    status = "evaluated"
                    clip_s = $clipSAvg
                    lpips = $lpipsAvg
                    n_pairs = $n
                    musiq = $null
                    note = "MUSIQ computed in Phase 3"
                }
            } else {
                "  [$mName] metrics.csv has no valid rows" | Tee-Object -FilePath $LOG -Append
                $results["phase1_wiki20"][$mName] = @{ status = "metrics empty"; musiq = $null; clip_s = $null; lpips = $null }
            }
        } catch {
            "  [$mName] Failed to parse metrics.csv: $_" | Tee-Object -FilePath $LOG -Append
            $results["phase1_wiki20"][$mName] = @{ status = "parse error"; error = "$_" }
        }
    } else {
        "  [$mName] metrics.csv not found" | Tee-Object -FilePath $LOG -Append
        $results["phase1_wiki20"][$mName] = @{ status = "no metrics.csv"; musiq = $null }
    }
    Save-Results $results
}

# ═══════════════════════════════════════════════════════════════
# Step 5: MUSIQ for Phase 1 methods (inline, via _compute_musiq_batch.py)
# ═══════════════════════════════════════════════════════════════
"--- Step 5: MUSIQ computation for Phase 1 methods ---" | Tee-Object -FilePath $LOG -Append

$musiqMethods = @()
foreach ($m in $methodsToEval) {
    $imgDir = "$($m.dir)\images"
    if (Test-Path $imgDir) {
        $cnt = (Get-ChildItem $imgDir -Filter "*.png" -ErrorAction SilentlyContinue).Count
        if ($cnt -gt 0) {
            $musiqMethods += "$($m.name)=$imgDir"
        }
    }
}

if ($musiqMethods.Count -gt 0) {
    $musiqArgsStr = $musiqMethods -join ","
    $musiqArgs = @(
        "-u",
        "$SCRIPTS_DIR\_compute_musiq_batch.py",
        "--methods", $musiqArgsStr,
        "--output", $RESULTS_JSON,
        "--key-suffix", "_wiki20",
        "--batch-size", "8"
    )
    $ec = Invoke-PythonTask "musiq-phase1" $musiqArgs $REPO "phase1.musiq"

    # Reload results (musiq script merges into the JSON)
    $results = Load-Results
} else {
    "  [musiq] No method directories with images found" | Tee-Object -FilePath $LOG -Append
}

Save-Results $results

"=== Phase 1 finished at $(Get-Date) ===" | Tee-Object -FilePath $LOG -Append
"  Results: $RESULTS_JSON" | Tee-Object -FilePath $LOG -Append
