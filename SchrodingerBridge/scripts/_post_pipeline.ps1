# Post-pipeline: SD-Turbo 256 gen + SaMam Wiki20 gen + re-eval ALL methods
# (Main pipeline evals failed due to CLIP API change; this script uses the fixed eval)
$ErrorActionPreference = "Continue"

$REPO = "I:\Github\Latent_Style\SchrodingerBridge"
$SCRIPTS_DIR = "$REPO\scripts"
$PYTHON = "C:\Program Files\Python312\python.exe"
$LOG = "$REPO\logs\post_pipeline.log"
$RESULTS_JSON = "$REPO\exp\_pipeline_fill_results.json"

# Env (must match pipeline_fill_main.ps1 for SYSTEM account compatibility)
$USER_SITE = "C:\Users\Administrator\AppData\Roaming\Python\Python312\site-packages"
$SRC_DIR = "$REPO\src"
$env:HF_HOME = "C:\Users\Administrator\.cache\huggingface"
$env:TRANSFORMERS_OFFLINE = "1"
$env:TORCH_HOME = "C:\Users\Administrator\.cache\torch"
$env:PYTHONPATH = "$SRC_DIR;$USER_SITE;$SCRIPTS_DIR"
$env:PYTHONUSERBASE = "C:\Users\Administrator\AppData\Roaming\Python"
$env:CUDA_VISIBLE_DEVICES = "0"

# Helpers
function Count-Images($dir) {
    if (-not (Test-Path $dir)) { return 0 }
    return (Get-ChildItem $dir -Filter *.png -ErrorAction SilentlyContinue).Count +
           (Get-ChildItem $dir -Filter *.jpg -ErrorAction SilentlyContinue).Count
}

function Save-Results($results) {
    $results | ConvertTo-Json -Depth 10 | Set-Content $RESULTS_JSON -Encoding UTF8
}

function Invoke-PythonTask($name, $pyArgs, $cwd, $logPrefix) {
    $cmd = "`"$PYTHON`" $($pyArgs -join ' ')"
    Write-Host "  CMD: $cmd"
    "  [$name] CMD: $cmd" | Out-File -FilePath $LOG -Append -Encoding UTF8
    $stderrFile = "$REPO\logs\$logPrefix.stderr.log"
    $stdoutFile = "$REPO\logs\$logPrefix.stdout.log"

    $proc = Start-Process -FilePath $PYTHON -ArgumentList $pyArgs -WorkingDirectory $cwd `
        -NoNewWindow -PassThru -RedirectStandardError $stderrFile -RedirectStandardOutput $stdoutFile
    $proc.WaitForExit()

    if ($proc.ExitCode -ne 0) {
        "  [$name] FAIL exit=$($proc.ExitCode) at $(Get-Date)" | Out-File -FilePath $LOG -Append -Encoding UTF8
        $stderrTail = Get-Content $stderrFile -Tail 25 -ErrorAction SilentlyContinue
        "    last 25 lines of stderr:" | Out-File -FilePath $LOG -Append -Encoding UTF8
        $stderrTail | Out-File -FilePath $LOG -Append -Encoding UTF8
    } else {
        "  [$name] OK exit=0 at $(Get-Date)" | Out-File -FilePath $LOG -Append -Encoding UTF8
    }
    return $proc.ExitCode
}

function Invoke-Eval($name, $imageDir, $dataset, $maxImages, $logPrefix) {
    $evalOut = "$REPO\exp\_eval_${name}.json"
    $pyArgs = @("-u", "$SCRIPTS_DIR\_eval_unified.py",
              "--image-dir", $imageDir,
              "--dataset", $dataset,
              "--output", $evalOut,
              "--max-images", "$maxImages")
    $code = Invoke-PythonTask "eval-$name" $pyArgs $REPO "eval_$logPrefix"
    if ($code -eq 0 -and (Test-Path $evalOut)) {
        return Get-Content $evalOut -Raw | ConvertFrom-Json
    }
    return $null
}

# Load existing results
$results = @{}
if (Test-Path $RESULTS_JSON) {
    try { $results = Get-Content $RESULTS_JSON -Raw | ConvertFrom-Json -AsHashtable } catch { $results = @{} }
}

"=" * 80 | Out-File -FilePath $LOG -Append -Encoding UTF8
"=== Post-pipeline started at $(Get-Date) ===" | Out-File -FilePath $LOG -Append -Encoding UTF8
"=" * 80 | Out-File -FilePath $LOG -Append -Encoding UTF8

# ── Phase 1: SD-Turbo 256 generation (fixed config: strength=0.5, steps=4) ──
"--- Phase 1: SD-Turbo 256 generation ---" | Out-File -FilePath $LOG -Append -Encoding UTF8
$sdturbo256Dir = "I:\exp_256_photo2art\sdturbo_256\images"
New-Item -ItemType Directory -Force -Path $sdturbo256Dir | Out-Null
$cnt = Count-Images $sdturbo256Dir
"  [sdturbo_256] existing: $cnt/750" | Out-File -FilePath $LOG -Append -Encoding UTF8
if ($cnt -lt 750) {
    $pyArgs = @("-u", "$SCRIPTS_DIR\_gen_diffusion_baseline.py",
              "--method", "sdturbo", "--test-dir", "I:\datasets\legacy256_overfit50\test",
              "--output-dir", $sdturbo256Dir, "--styles", "cezanne,Hayao,monet,photo,vangogh",
              "--image-size", "256", "--max-src-per-style", "30")
    Invoke-PythonTask "sdturbo_256" $pyArgs $REPO "sdturbo256.gen" | Out-Null
}
$cnt = Count-Images $sdturbo256Dir
$results["sdturbo_256_gen_count"] = $cnt
"  [sdturbo_256] final: $cnt/750" | Out-File -FilePath $LOG -Append -Encoding UTF8
Save-Results $results

# ── Phase 2: SaMam Wiki20 generation ──
"--- Phase 2: SaMam Wiki20 generation ---" | Out-File -FilePath $LOG -Append -Encoding UTF8
$samamW20Dir = "$REPO\exp\baseline_wikiarts20\samam\images"
New-Item -ItemType Directory -Force -Path $samamW20Dir | Out-Null
$cnt = Count-Images $samamW20Dir
"  [samam_w20] existing: $cnt/750" | Out-File -FilePath $LOG -Append -Encoding UTF8
if ($cnt -lt 750) {
    $pyArgs = @("-u", "$SCRIPTS_DIR\_gen_samam_wiki20.py")
    Invoke-PythonTask "samam_w20" $pyArgs $REPO "samamw20.gen" | Out-Null
}
$cnt = Count-Images $samamW20Dir
$results["samam_w20_gen_count"] = $cnt
"  [samam_w20] final: $cnt/750" | Out-File -FilePath $LOG -Append -Encoding UTF8
Save-Results $results

# ── Phase 3: Re-evaluate ALL methods with fixed eval script ──
"--- Phase 3: Re-evaluate ALL methods (fixed CLIP API) ---" | Out-File -FilePath $LOG -Append -Encoding UTF8
$styleid256Dir = "I:\exp_256_photo2art\styleid_256\images"
$samstW20Dir = "$REPO\exp\baseline_wikiarts20\samst\images"
$sdturboW20Dir = "$REPO\exp\baseline_wikiarts20\sdturbo\images"
$styleidW20Dir = "$REPO\exp\baseline_wikiarts20\styleid\images"

$evalTargets = @(
    @{ name = "sdturbo_256"; dir = $sdturbo256Dir; dataset = "photo2art256"; max = 750 },
    @{ name = "styleid_256"; dir = $styleid256Dir; dataset = "photo2art256"; max = 750 },
    @{ name = "samst_w20";   dir = $samstW20Dir;   dataset = "wiki20distinct5"; max = 750 },
    @{ name = "sdturbo_w20"; dir = $sdturboW20Dir; dataset = "wiki20distinct5"; max = 750 },
    @{ name = "styleid_w20"; dir = $styleidW20Dir; dataset = "wiki20distinct5"; max = 750 },
    @{ name = "samam_w20";   dir = $samamW20Dir;   dataset = "wiki20distinct5"; max = 750 }
)

foreach ($t in $evalTargets) {
    $cnt = Count-Images $t.dir
    if ($cnt -eq 0) {
        "  [eval-$($t.name)] SKIP: 0 images" | Out-File -FilePath $LOG -Append -Encoding UTF8
        continue
    }
    "  [eval-$($t.name)] Evaluating $cnt images ($($t.dataset))..." | Out-File -FilePath $LOG -Append -Encoding UTF8
    $evalResult = Invoke-Eval $t.name $t.dir $t.dataset $t.max $t.name
    if ($evalResult) {
        $results[$t.name] = $evalResult
        Save-Results $results
        "  [eval-$($t.name)] CLIP-S=$($evalResult.clip_s)  LPIPS=$($evalResult.lpips)  MUSIQ=$($evalResult.musiq)" | Out-File -FilePath $LOG -Append -Encoding UTF8
    } else {
        "  [eval-$($t.name)] FAILED" | Out-File -FilePath $LOG -Append -Encoding UTF8
    }
}

# Done
"=" * 80 | Out-File -FilePath $LOG -Append -Encoding UTF8
"=== Post-pipeline COMPLETED at $(Get-Date) ===" | Out-File -FilePath $LOG -Append -Encoding UTF8
"=" * 80 | Out-File -FilePath $LOG -Append -Encoding UTF8
Save-Results $results
Write-Host "Post-pipeline completed. Results: $RESULTS_JSON"
