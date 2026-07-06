# Phase 3: MUSIQ re-run for 512 Distinct5 + WikiArt-20
# Re-evaluates existing baseline images with MUSIQ enabled.
#
# - 512 Distinct5: existing exp\baseline_v2\images\{adain,cut,identity,samam,samst,sdturbo,styleid,...}\*.png (750 each)
# - WikiArt-20: existing exp\baseline_wikiarts20\{identity,adain,wct,samam,sdturbo}\images\*.png (12000 each)
#               + exp\wikiarts20_eval\images\*.png (WEAVE)
#
# Uses _compute_musiq_batch.py for efficient batch MUSIQ computation.
# Logs to logs\phase3_musiq_rerun.log
# Aggregated results written to exp\_baseline_fill_results.json

$ErrorActionPreference = "Continue"

# ── Paths ──
$REPO = "I:\Github\Latent_Style\SchrodingerBridge"
$PYTHON = "C:\Program Files\Python312\python.exe"
$SRC_DIR = "$REPO\src"
$SCRIPTS_DIR = "$REPO\scripts"
$LOG_DIR = "$REPO\logs"
$LOG = "$LOG_DIR\phase3_musiq_rerun.log"
$RESULTS_JSON = "$REPO\exp\_baseline_fill_results.json"

# 512 Distinct5 baseline images
$BASELINE_V2 = "$REPO\exp\baseline_v2\images"

# WikiArt-20 baseline images
$WIKI20_ROOT = "$REPO\exp\baseline_wikiarts20"
$WIKI20_WEAVE = "$REPO\exp\wikiarts20_eval\images"

# ── Environment ──
$USER_SITE = "C:\Users\Administrator\AppData\Roaming\Python\Python312\site-packages"
$env:PYTHONPATH = "$SRC_DIR;$USER_SITE"
$env:PYTHONUSERBASE = "C:\Users\Administrator\AppData\Roaming\Python"
$env:CUDA_VISIBLE_DEVICES = "0"
$env:HF_HOME = "$REPO\exp\eval_cache\hf"
$env:TRANSFORMERS_OFFLINE = "0"

# ── Setup ──
New-Item -ItemType Directory -Force -Path $LOG_DIR | Out-Null
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

# Helper: collect method=image_dir pairs from a parent directory
function Collect-MethodDirs($parentDir, $imageSubdir, $prefix) {
    $pairs = @()
    if (-not (Test-Path $parentDir)) {
        "  [collect] Parent dir not found: $parentDir" | Tee-Object -FilePath $LOG -Append
        return $pairs
    }
    $subdirs = Get-ChildItem $parentDir -Directory -ErrorAction SilentlyContinue
    foreach ($sd in $subdirs) {
        $imgDir = if ($imageSubdir) { Join-Path $sd.FullName $imageSubdir } else { $sd.FullName }
        if (Test-Path $imgDir) {
            $cnt = (Get-ChildItem $imgDir -Filter "*.png" -ErrorAction SilentlyContinue).Count
            $cntJpg = (Get-ChildItem $imgDir -Filter "*.jpg" -ErrorAction SilentlyContinue).Count
            $total = $cnt + $cntJpg
            if ($total -gt 0) {
                $name = if ($prefix) { "${prefix}_$($sd.Name)" } else { $sd.Name }
                $pairs += @{ name = $name; dir = $imgDir; count = $total }
                "  [collect] $name : $total images ($imgDir)" | Tee-Object -FilePath $LOG -Append
            }
        }
    }
    return $pairs
}

$results = Load-Results
if (-not $results.ContainsKey("phase3_musiq")) {
    $results["phase3_musiq"] = @{}
}

"=== Phase 3: MUSIQ re-run (512 Distinct5 + WikiArt-20) started at $(Get-Date) ===" | Tee-Object -FilePath $LOG -Append
"  baseline_v2 (512 distinct5): $BASELINE_V2" | Tee-Object -FilePath $LOG -Append
"  wiki20_root: $WIKI20_ROOT" | Tee-Object -FilePath $LOG -Append
"  wiki20_weave: $WIKI20_WEAVE" | Tee-Object -FilePath $LOG -Append

# ═══════════════════════════════════════════════════════════════
# Step 1: Collect all method directories
# ═══════════════════════════════════════════════════════════════
"--- Step 1: Collecting method directories ---" | Tee-Object -FilePath $LOG -Append

$allPairs = @()

# 512 Distinct5: baseline_v2/images/{method}/*.png (each method is a subdir with PNGs directly)
"  [512 distinct5] Scanning $BASELINE_V2 ..." | Tee-Object -FilePath $LOG -Append
$distinct5Pairs = Collect-MethodDirs $BASELINE_V2 "" "512"
$allPairs += $distinct5Pairs

# WikiArt-20: baseline_wikiarts20/{method}/images/*.png
"  [wiki20] Scanning $WIKI20_ROOT ..." | Tee-Object -FilePath $LOG -Append
$wiki20Pairs = Collect-MethodDirs $WIKI20_ROOT "images" "wiki20"
$allPairs += $wiki20Pairs

# WikiArt-20 WEAVE: wikiarts20_eval/images/*.png (single directory)
if (Test-Path $WIKI20_WEAVE) {
    $cnt = (Get-ChildItem $WIKI20_WEAVE -Filter "*.png" -ErrorAction SilentlyContinue).Count
    if ($cnt -gt 0) {
        "  [collect] wiki20_weave : $cnt images ($WIKI20_WEAVE)" | Tee-Object -FilePath $LOG -Append
        $allPairs += @{ name = "wiki20_weave"; dir = $WIKI20_WEAVE; count = $cnt }
    }
}

"  Total methods to evaluate: $($allPairs.Count)" | Tee-Object -FilePath $LOG -Append

# ═══════════════════════════════════════════════════════════════
# Step 2: Run MUSIQ batch computation
# ═══════════════════════════════════════════════════════════════
"--- Step 2: MUSIQ batch computation ---" | Tee-Object -FilePath $LOG -Append

if ($allPairs.Count -eq 0) {
    "  [musiq] No method directories found. Skipping." | Tee-Object -FilePath $LOG -Append
    $results["phase3_musiq"]["status"] = "no methods found"
    Save-Results $results
} else {
    # Build --methods argument string
    $methodsStr = ($allPairs | ForEach-Object { "$($_.name)=$($_.dir)" }) -join ","

    # Run in batches if the methods string is very long (to avoid command-line length limits)
    # Split into chunks of ~10 methods
    $batchSize = 10
    $batches = @()
    for ($i = 0; $i -lt $allPairs.Count; $i += $batchSize) {
        $end = [Math]::Min($i + $batchSize, $allPairs.Count)
        $batch = $allPairs[$i..($end - 1)]
        $batches += ,@($batch)
    }

    "  Split into $($batches.Count) batch(es) of up to $batchSize methods each" | Tee-Object -FilePath $LOG -Append

    $batchNum = 0
    foreach ($batch in $batches) {
        $batchNum++
        $batchStr = ($batch | ForEach-Object { "$($_.name)=$($_.dir)" }) -join ","
        "  Batch $batchNum/$($batches.Count): $($batch.Count) methods" | Tee-Object -FilePath $LOG -Append

        $musiqArgs = @(
            "-u",
            "$SCRIPTS_DIR\_compute_musiq_batch.py",
            "--methods", $batchStr,
            "--output", $RESULTS_JSON,
            "--batch-size", "8"
        )

        try {
            $ec = Invoke-PythonTask "musiq-batch-$batchNum" $musiqArgs $REPO "musiq.batch$batchNum"
            # Reload results (musiq script merges into JSON)
            $results = Load-Results
        } catch {
            "  [musiq-batch-$batchNum] EXCEPTION: $_ at $(Get-Date)" | Tee-Object -FilePath $LOG -Append
        }
    }

    # Ensure phase3_musiq key exists with summary
    $results = Load-Results
    if (-not $results.ContainsKey("phase3_musiq")) {
        $results["phase3_musiq"] = @{}
    }
    $results["phase3_musiq"]["n_methods"] = $allPairs.Count
    $results["phase3_musiq"]["methods_processed"] = ($allPairs | ForEach-Object { $_.name }) -join ", "
    Save-Results $results
}

# ═══════════════════════════════════════════════════════════════
# Step 3: Summary — aggregate MUSIQ scores per method
# ═══════════════════════════════════════════════════════════════
"--- Step 3: Summary ---" | Tee-Object -FilePath $LOG -Append

$results = Load-Results
if (-not $results.ContainsKey("phase3_musiq")) {
    $results["phase3_musiq"] = @{}
}
$musiqSummary = @{}

# Collect all MUSIQ scores from top-level keys (the _compute_musiq_batch.py writes flat keys)
foreach ($key in $results.Keys) {
    if ($key -eq "phase1_wiki20" -or $key -eq "phase2_256" -or $key -eq "phase3_musiq") {
        continue
    }
    $val = $results[$key]
    if ($val -is [hashtable] -and $val.ContainsKey("musiq")) {
        $musiqSummary[$key] = @{
            musiq = $val["musiq"]
            n_images = $val["n_images"]
        }
    }
}

# Also check phase1/phase2 sub-methods
foreach ($phaseKey in @("phase1_wiki20", "phase2_256")) {
    if ($results.ContainsKey($phaseKey)) {
        $phaseData = $results[$phaseKey]
        if ($phaseData -is [hashtable]) {
            foreach ($mk in $phaseData.Keys) {
                $mv = $phaseData[$mk]
                if ($mv -is [hashtable] -and $mv.ContainsKey("musiq") -and $null -ne $mv["musiq"]) {
                    $musiqSummary["${phaseKey}_${mk}"] = @{
                        musiq = $mv["musiq"]
                        n_images = $mv["n_images"]
                    }
                }
            }
        }
    }
}

$results["phase3_musiq"]["summary"] = $musiqSummary
Save-Results $results

"  MUSIQ scores (top-level + phase methods):" | Tee-Object -FilePath $LOG -Append
foreach ($key in ($musiqSummary.Keys | Sort-Object)) {
    $v = $musiqSummary[$key]
    $musiqStr = if ($null -ne $v["musiq"]) { "{0:N4}" -f $v["musiq"] } else { "None" }
    "    $key : MUSIQ=$musiqStr  (n=$($v['n_images']))" | Tee-Object -FilePath $LOG -Append
}

"=== Phase 3 finished at $(Get-Date) ===" | Tee-Object -FilePath $LOG -Append
"  Results: $RESULTS_JSON" | Tee-Object -FilePath $LOG -Append
"  Total methods with MUSIQ: $($musiqSummary.Count)" | Tee-Object -FilePath $LOG -Append
