# T11 evolution batch: sequential train+eval on local GPU (RTX 4070 8GB)
# Records training and eval time for each experiment
$ErrorActionPreference = "Continue"
Set-Location "G:\GitHub\Latent_Style\SchrodingerBridge"
$env:PYTHONPATH = "."
$env:PYTHONIOENCODING = "utf-8"

$testDir = "G:\GitHub\Latent_Style\Dataset\distinct5_512\test"
$cacheDir = "G:\GitHub\Latent_Style\SchrodingerBridge\exp\eval_cache"
$hfCache = "G:\GitHub\Latent_Style\SchrodingerBridge\exp\eval_cache\hf"

# Experiments to run (order: most promising first)
$exps = @(
    @{name="t11e_a_hf_boost"; epochs=10; finalEp="epoch_0010"},
    @{name="t11e_b_ll_leak"; epochs=10; finalEp="epoch_0010"},
    @{name="t11e_c_aggressive"; epochs=10; finalEp="epoch_0010"},
    @{name="t11e_d_pure_dwt_long"; epochs=15; finalEp="epoch_0015"}
)

$resultsFile = "G:\GitHub\Latent_Style\SchrodingerBridge\exp\t11evo\timing_results.txt"
"=== T11 Evolution Batch Start $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') ===" | Out-File $resultsFile -Encoding utf8

foreach ($expInfo in $exps) {
    $exp = $expInfo.name
    $config = "configs\$exp.json"
    $ckpt = "exp\t11evo\$exp\$($expInfo.finalEp).pt"
    $evalDir = "exp\t11evo\$exp\full_eval\$($expInfo.finalEp)"
    $logOut = "G:\GitHub\Latent_Style\SchrodingerBridge\exp\t11evo\${exp}_train_eval.log"

    Write-Output "=== EXP=$exp START $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') ==="
    "=== EXP=$exp START $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') ===" | Out-File $resultsFile -Append -Encoding utf8

    # TRAIN
    $trainStart = Get-Date
    Write-Output "=== TRAIN START $trainStart ==="
    python -u src\run.py --config $config 2>&1 | Tee-Object -FilePath $logOut
    $trainEc = $LASTEXITCODE
    $trainEnd = Get-Date
    $trainDur = $trainEnd - $trainStart
    $trainMin = [math]::Round($trainDur.TotalMinutes, 1)
    Write-Output "=== TRAIN DONE exit=$trainEc duration=${trainMin}min ($trainEnd) ==="
    "TRAIN: exit=$trainEc duration=${trainMin}min start=$trainStart end=$trainEnd" | Out-File $resultsFile -Append -Encoding utf8

    if ($trainEc -ne 0 -or -not (Test-Path $ckpt)) {
        # Try to find any checkpoint
        $anyCkpt = Get-ChildItem "exp\t11evo\$exp" -Filter "*.pt" -ErrorAction SilentlyContinue | Sort-Object Name -Descending | Select-Object -First 1
        if ($anyCkpt) {
            $ckpt = $anyCkpt.FullName
            $evalDir = "exp\t11evo\$exp\full_eval\$($anyCkpt.BaseName)"
            Write-Output "Using checkpoint: $ckpt"
        } else {
            Write-Output "FATAL: no checkpoint for $exp. Skipping."
            "FATAL: no checkpoint for $exp. SKIP." | Out-File $resultsFile -Append -Encoding utf8
            continue
        }
    }

    # EVAL (clip-style + lpips only)
    $evalStart = Get-Date
    Write-Output "=== EVAL START $evalStart ==="
    python -u src\utils\run_evaluation.py `
        --checkpoint $ckpt `
        --output $evalDir `
        --test_dir $testDir `
        --cache_dir $cacheDir `
        --clip_hf_cache_dir $hfCache `
        --batch_size 2 --generation_batch_size 2 --metric_batch_size 2 `
        --target_chunk_size 2 --vae_decode_batch_size 16 `
        --eval_only_lpips_clip_style --eval_lpips_chunk_size 4 2>&1 | Tee-Object -FilePath $logOut -Append
    $evalEc = $LASTEXITCODE
    $evalEnd = Get-Date
    $evalDur = $evalEnd - $evalStart
    $evalMin = [math]::Round($evalDur.TotalMinutes, 1)
    Write-Output "=== EVAL DONE exit=$evalEc duration=${evalMin}min ($evalEnd) ==="
    "EVAL: exit=$evalEc duration=${evalMin}min start=$evalStart end=$evalEnd" | Out-File $resultsFile -Append -Encoding utf8

    # Extract metrics
    $summary = Join-Path $evalDir "summary.json"
    if (Test-Path $summary) {
        $metrics = python -c "import json; d=json.load(open(r'$summary','r',encoding='utf-8')); apo=d.get('analysis',{}).get('all_pairs_overview',{}); print(f\"clip_style={apo.get('clip_style','N/A')}, content_lpips={apo.get('content_lpips','N/A')}\")"
        Write-Output "METRICS: $metrics"
        "METRICS: $metrics" | Out-File $resultsFile -Append -Encoding utf8
    }

    Write-Output "=== EXP=$exp COMPLETE $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') ==="
    "" | Out-File $resultsFile -Append -Encoding utf8
}

Write-Output "=== ALL T11 EVO BATCH COMPLETE $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') ==="
"=== ALL T11 EVO BATCH COMPLETE $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') ===" | Out-File $resultsFile -Append -Encoding utf8
Get-Content $resultsFile
