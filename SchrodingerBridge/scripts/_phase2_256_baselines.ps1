# Phase 2: 256 baselines — SD-Turbo, StyleID, CUT + seedream LPIPS
# Generates 750 images per method (5 styles x 5 styles x 30 src) on legacy256_overfit50\test,
# then evaluates each with MUSIQ + CLIP-S + LPIPS.
#
# - sdturbo_256: SD-Turbo img2img (training-free). Output to I:\exp_256_photo2art\sdturbo_256\
# - styleid_256: SD1.5 img2img (training-free). Output to I:\exp_256_photo2art\styleid_256\
# - cut_256: needs per-style training. Skipped with "--" note if no checkpoints.
# - seedream_256 LPIPS: re-evaluate existing seedream_256 images with LPIPS.
# - Evaluate all with MUSIQ + CLIP-S + LPIPS via batch_compute_photo2art.py.
#
# Skip-resumable: existing output PNGs are skipped.
# Logs to logs\phase2_256_baselines.log
# Aggregated results written to exp\_baseline_fill_results.json

$ErrorActionPreference = "Continue"

# ── Paths ──
$REPO = "I:\Github\Latent_Style\SchrodingerBridge"
$PYTHON = "C:\Program Files\Python312\python.exe"
$SRC_DIR = "$REPO\src"
$SCRIPTS_DIR = "$REPO\scripts"
$TEST_DIR = "I:\datasets\legacy256_overfit50\test"
$EXP_256 = "I:\exp_256_photo2art"
$LOG_DIR = "$REPO\logs"
$LOG = "$LOG_DIR\phase2_256_baselines.log"
$RESULTS_JSON = "$REPO\exp\_baseline_fill_results.json"

# CUT paths
$CUT_REPO = "I:\Github\Latent_Style\Related_Works\repos\external\CUT"

# ── Environment ──
$USER_SITE = "C:\Users\Administrator\AppData\Roaming\Python\Python312\site-packages"
$env:PYTHONPATH = "$SRC_DIR;$USER_SITE;$SCRIPTS_DIR"
$env:PYTHONUSERBASE = "C:\Users\Administrator\AppData\Roaming\Python"
$env:CUDA_VISIBLE_DEVICES = "0"
$env:HF_HOME = "$REPO\exp\eval_cache\hf"
$env:TRANSFORMERS_OFFLINE = "0"

# ── Legacy 256 styles ──
$LEGACY5 = "cezanne,Hayao,monet,photo,vangogh"

# ── Setup ──
New-Item -ItemType Directory -Force -Path $LOG_DIR | Out-Null
New-Item -ItemType Directory -Force -Path $EXP_256 | Out-Null
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
if (-not $results.ContainsKey("phase2_256")) {
    $results["phase2_256"] = @{}
}

"=== Phase 2: 256 baselines (SD-Turbo/StyleID/CUT) + seedream LPIPS started at $(Get-Date) ===" | Tee-Object -FilePath $LOG -Append
"  test_dir: $TEST_DIR" | Tee-Object -FilePath $LOG -Append
"  exp_256: $EXP_256" | Tee-Object -FilePath $LOG -Append
"  styles: $LEGACY5" | Tee-Object -FilePath $LOG -Append

# ═══════════════════════════════════════════════════════════════
# Step 1: SD-Turbo 256 — 750 images
# ═══════════════════════════════════════════════════════════════
"--- Step 1: SD-Turbo 256 (750 images) ---" | Tee-Object -FilePath $LOG -Append

try {
    $sdturboDir = "$EXP_256\sdturbo_256"
    $sdturboImages = "$sdturboDir\images"
    New-Item -ItemType Directory -Force -Path $sdturboImages | Out-Null

    $sdturboArgs = @(
        "-u",
        "$SCRIPTS_DIR\_gen_diffusion_baseline.py",
        "--method", "sdturbo",
        "--test-dir", $TEST_DIR,
        "--output-dir", $sdturboImages,
        "--styles", $LEGACY5,
        "--image-size", "256",
        "--max-src-per-style", "30"
    )

    $ec = Invoke-PythonTask "sdturbo-gen" $sdturboArgs $REPO "sdturbo.gen"
    $imgCount = 0
    if (Test-Path $sdturboImages) {
        $imgCount = (Get-ChildItem $sdturboImages -Filter "*.png" -ErrorAction SilentlyContinue).Count
    }
    "  [sdturbo_256] image count: $imgCount (expected 750)" | Tee-Object -FilePath $LOG -Append
    $results["phase2_256"]["sdturbo_256_gen_count"] = $imgCount
    Save-Results $results
} catch {
    "  [sdturbo_256] GEN EXCEPTION: $_ at $(Get-Date)" | Tee-Object -FilePath $LOG -Append
    $results["phase2_256"]["sdturbo_256_error"] = "$_"
    Save-Results $results
}

# ═══════════════════════════════════════════════════════════════
# Step 2: StyleID 256 — 750 images
# ═══════════════════════════════════════════════════════════════
"--- Step 2: StyleID 256 (750 images) ---" | Tee-Object -FilePath $LOG -Append

try {
    $styleidDir = "$EXP_256\styleid_256"
    $styleidImages = "$styleidDir\images"
    New-Item -ItemType Directory -Force -Path $styleidImages | Out-Null

    $styleidArgs = @(
        "-u",
        "$SCRIPTS_DIR\_gen_diffusion_baseline.py",
        "--method", "styleid",
        "--test-dir", $TEST_DIR,
        "--output-dir", $styleidImages,
        "--styles", $LEGACY5,
        "--image-size", "256",
        "--max-src-per-style", "30"
    )

    $ec = Invoke-PythonTask "styleid-gen" $styleidArgs $REPO "styleid.gen"
    $imgCount = 0
    if (Test-Path $styleidImages) {
        $imgCount = (Get-ChildItem $styleidImages -Filter "*.png" -ErrorAction SilentlyContinue).Count
    }
    "  [styleid_256] image count: $imgCount (expected 750)" | Tee-Object -FilePath $LOG -Append
    $results["phase2_256"]["styleid_256_gen_count"] = $imgCount
    Save-Results $results
} catch {
    "  [styleid_256] GEN EXCEPTION: $_ at $(Get-Date)" | Tee-Object -FilePath $LOG -Append
    $results["phase2_256"]["styleid_256_error"] = "$_"
    Save-Results $results
}

# ═══════════════════════════════════════════════════════════════
# Step 3: CUT 256 — needs per-style training; skip with note if no checkpoints
# ═══════════════════════════════════════════════════════════════
"--- Step 3: CUT 256 (per-style training required) ---" | Tee-Object -FilePath $LOG -Append

try {
    $cutCkptDir = "$CUT_REPO\checkpoints"
    $cutCkpts = @()
    if (Test-Path $cutCkptDir) {
        $cutCkpts = Get-ChildItem $cutCkptDir -Recurse -Filter "*.pth" -ErrorAction SilentlyContinue
    }

    # Also check CUT full_eval for existing results
    $cutFullEval = "$CUT_REPO\full_eval"
    if ($cutCkpts.Count -eq 0 -and (Test-Path $cutFullEval)) {
        "  [cut_256] No checkpoints but full_eval dir exists. Checking for existing results..." | Tee-Object -FilePath $LOG -Append
    }

    if ($cutCkpts.Count -eq 0) {
        "  [cut_256] No pretrained checkpoints found at $cutCkptDir" | Tee-Object -FilePath $LOG -Append
        "  [cut_256] CUT requires per-style training. Marking as '--'." | Tee-Object -FilePath $LOG -Append
        $results["phase2_256"]["cut_256"] = @{ status = "skipped"; reason = "no pretrained checkpoints; per-style training required"; musiq = $null; clip_s = $null; lpips = $null }
        Save-Results $results
    } else {
        "  [cut_256] Found $($cutCkpts.Count) checkpoint(s)." | Tee-Object -FilePath $LOG -Append
        $results["phase2_256"]["cut_256"] = @{ status = "partial"; reason = "checkpoints exist but CUT inference not automated in this script"; n_ckpts = $cutCkpts.Count }
        Save-Results $results
    }
} catch {
    "  [cut_256] EXCEPTION: $_ at $(Get-Date)" | Tee-Object -FilePath $LOG -Append
    $results["phase2_256"]["cut_256"] = @{ status = "error"; error = "$_" }
    Save-Results $results
}

# ═══════════════════════════════════════════════════════════════
# Step 4: seedream_256 LPIPS — re-evaluate with LPIPS
# ═══════════════════════════════════════════════════════════════
"--- Step 4: seedream_256 LPIPS re-evaluation ---" | Tee-Object -FilePath $LOG -Append

try {
    # Check seedream_256 images (from methods_256_photo2art.json: /mnt/i/Github/Latent_Style/seedream45_api/protocol_a_800/images)
    $seedreamDir = "I:\Github\Latent_Style\seedream45_api\protocol_a_800"
    $seedreamImages = "$seedreamDir\images"

    if (-not (Test-Path $seedreamImages)) {
        # Try alternate location
        $seedreamDir = "$EXP_256\seedream_256"
        $seedreamImages = "$seedreamDir\images"
    }

    if (Test-Path $seedreamImages) {
        $imgCount = (Get-ChildItem $seedreamImages -Filter "*.png" -ErrorAction SilentlyContinue).Count
        $imgCountJpg = (Get-ChildItem $seedreamImages -Filter "*.jpg" -ErrorAction SilentlyContinue).Count
        $total = $imgCount + $imgCountJpg
        "  [seedream_256] image count: $total ($imgCount png + $imgCountJpg jpg)" | Tee-Object -FilePath $LOG -Append

        if ($total -gt 0) {
            # Re-evaluate with LPIPS via run_evaluation.py --reuse_generated
            $seedreamArgs = @(
                "-u",
                "utils\run_evaluation.py",
                $seedreamDir,
                "--reuse_generated",
                "--style_subdirs", $LEGACY5,
                "--test_dir", $TEST_DIR,
                "--eval_only_lpips_clip_style",
                "--max_src_samples", "30",
                "--batch_size", "2",
                "--ref_feature_batch_size", "2"
            )

            $ec = Invoke-PythonTask "seedream-lpips" $seedreamArgs $SRC_DIR "seedream.lpips"

            # Read metrics.csv
            $metricsCsv = "$seedreamDir\metrics.csv"
            if (Test-Path $metricsCsv) {
                try {
                    $csvContent = Import-Csv $metricsCsv
                    $lpipsSum = 0.0; $clipSSum = 0.0; $n = 0
                    foreach ($row in $csvContent) {
                        try {
                            $lpipsVal = [double]$row.content_lpips
                            $clipSVal = [double]$row.clip_style
                            $lpipsSum += $lpipsVal
                            $clipSSum += $clipSVal
                            $n++
                        } catch { }
                    }
                    if ($n -gt 0) {
                        $lpipsAvg = $lpipsSum / $n
                        $clipSAvg = $clipSSum / $n
                        "  [seedream_256] LPIPS=$lpipsAvg  CLIP-S=$clipSAvg  (n=$n)" | Tee-Object -FilePath $LOG -Append
                        $results["phase2_256"]["seedream_256_lpips"] = @{
                            status = "evaluated"
                            lpips = $lpipsAvg
                            clip_s = $clipSAvg
                            n_pairs = $n
                        }
                    }
                } catch {
                    "  [seedream_256] Failed to parse metrics.csv: $_" | Tee-Object -FilePath $LOG -Append
                }
            }
            Save-Results $results
        } else {
            "  [seedream_256] No images found. Skipping LPIPS re-eval." | Tee-Object -FilePath $LOG -Append
            $results["phase2_256"]["seedream_256_lpips"] = @{ status = "no images"; lpips = $null }
            Save-Results $results
        }
    } else {
        "  [seedream_256] Directory not found: $seedreamImages" | Tee-Object -FilePath $LOG -Append
        $results["phase2_256"]["seedream_256_lpips"] = @{ status = "dir not found"; path = $seedreamImages; lpips = $null }
        Save-Results $results
    }
} catch {
    "  [seedream_256] EXCEPTION: $_ at $(Get-Date)" | Tee-Object -FilePath $LOG -Append
    $results["phase2_256"]["seedream_256_lpips"] = @{ status = "error"; error = "$_" }
    Save-Results $results
}

# ═══════════════════════════════════════════════════════════════
# Step 5: Evaluate all 256 methods with MUSIQ + CLIP-S + LPIPS
# Uses batch_compute_photo2art.py (computes CLIP-S, CLIP-T, MUSIQ, ART-FID)
# ═══════════════════════════════════════════════════════════════
"--- Step 5: Evaluate 256 methods (MUSIQ + CLIP-S + LPIPS via batch_compute) ---" | Tee-Object -FilePath $LOG -Append

try {
    # Build methods JSON for batch_compute_photo2art.py
    $methodsJson = @{}
    $methodsList = @(
        @{ name = "sdturbo_256"; dir = "$EXP_256\sdturbo_256\images" },
        @{ name = "styleid_256"; dir = "$EXP_256\styleid_256\images" },
        @{ name = "seedream_256"; dir = "I:\Github\Latent_Style\seedream45_api\protocol_a_800\images" }
    )

    foreach ($m in $methodsList) {
        if (Test-Path $m.dir) {
            $cnt = (Get-ChildItem $m.dir -Filter "*.png" -ErrorAction SilentlyContinue).Count
            $cntJpg = (Get-ChildItem $m.dir -Filter "*.jpg" -ErrorAction SilentlyContinue).Count
            if (($cnt + $cntJpg) -gt 0) {
                $methodsJson[$m.name] = @{ gen_dir = $m.dir }
            }
        }
    }

    if ($methodsJson.Count -gt 0) {
        $methodsJsonPath = "$SCRIPTS_DIR\_methods_phase2_256.json"
        $methodsJson | ConvertTo-Json -Depth 3 | Out-File -FilePath $methodsJsonPath -Encoding utf8 -Force
        "  Methods JSON: $methodsJsonPath ($($methodsJson.Count) methods)" | Tee-Object -FilePath $LOG -Append

        $batchOut = "$EXP_256\eval_phase2_256.json"
        $batchArgs = @(
            "-u",
            "$SCRIPTS_DIR\batch_compute_photo2art.py",
            "--methods-json", $methodsJsonPath,
            "--output", $batchOut,
            "--max-images", "750",
            "--max-gen-artfid", "200"
        )

        $ec = Invoke-PythonTask "batch-eval-256" $batchArgs $SCRIPTS_DIR "batch256"

        # Read results
        if (Test-Path $batchOut) {
            try {
                $batchResults = Get-Content $batchOut -Raw | ConvertFrom-Json -AsHashtable
                foreach ($key in $batchResults.Keys) {
                    $results["phase2_256"][$key] = $batchResults[$key]
                }
                "  [batch-eval] Merged $($batchResults.Count) method results" | Tee-Object -FilePath $LOG -Append
            } catch {
                "  [batch-eval] Failed to parse ${batchOut}: $_" | Tee-Object -FilePath $LOG -Append
            }
        }
        Save-Results $results
    } else {
        "  [batch-eval] No method directories with images found" | Tee-Object -FilePath $LOG -Append
    }
} catch {
    "  [batch-eval] EXCEPTION: $_ at $(Get-Date)" | Tee-Object -FilePath $LOG -Append
    $results["phase2_256"]["batch_eval_error"] = "$_"
    Save-Results $results
}

"=== Phase 2 finished at $(Get-Date) ===" | Tee-Object -FilePath $LOG -Append
"  Results: $RESULTS_JSON" | Tee-Object -FilePath $LOG -Append
